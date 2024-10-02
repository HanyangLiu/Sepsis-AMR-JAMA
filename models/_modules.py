import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple
import torch.nn.utils.rnn as rnn_utils
from utils.types_ import *
from paths import comorb_vocab_size


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, embed_weights, embed_size, maxlen, use_pretrain, freeze_embed, num_layers, num_heads,
                 output_size, dropout_rate, device, pad_idx=0):
        super(Transformer, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        self.embed_size = embed_size
        self.maxlen = maxlen
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        if use_pretrain:
            self.embedder = nn.Embedding.from_pretrained(
                torch.FloatTensor(embed_weights),
                freeze=freeze_embed)
        else:
            self.embedder = nn.Embedding(
                num_embeddings=comorb_vocab_size + 2,
                embedding_dim=embed_size,
                padding_idx=0)

        self.dropout = nn.Dropout(dropout_rate)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, inputs, targets=None):
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S
        embeds = self.forward_to_embeds(inputs)

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # S x N x E
        outputs = encoded_inputs.permute(1, 0, 2)  # N x S x E
        pooled_outputs = outputs.mean(dim=1)

        return pooled_outputs

    def forward_to_embeds(self, inputs):
        inputs = inputs.to(torch.int64)
        embeds = self.embedder(inputs) * math.sqrt(self.embed_size)  # N x S x E
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # S x N x E
        embeds = embeds.to(self.device)

        return embeds


class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_size, num_modalities, num_layers, num_heads,
                 output_size, dropout_rate, device, pad_idx=0):
        super(SelfAttentionFusion, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        self.embed_size = embed_size
        self.num_modalities = num_modalities
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        self.dropout = nn.Dropout(dropout_rate)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, inputs, modality_mask, targets=None):
        src_key_padding_mask = modality_mask # N x S
        embeds = inputs.permute(1, 0, 2)  # S x N x E
        embeds = embeds.to(self.device)

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # S x N x E
        outputs = encoded_inputs.permute(1, 0, 2)  # N x S x E
        # pooled_outputs = torch.flatten(outputs, start_dim=1, end_dim=2)
        pooled_outputs = outputs.mean(dim=1)

        return pooled_outputs


class ProcedureTransformer(nn.Module):
    def __init__(self, embed_weights, embed_size, maxlen, use_pretrain, freeze_embed, num_layers, num_heads,
                 output_size, dropout_rate, device, pad_idx=0):
        super(ProcedureTransformer, self).__init__()
        if embed_size % num_heads != 0:
            raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
        self.embed_size = embed_size
        self.maxlen = maxlen
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        if use_pretrain:
            self.embedder = nn.Embedding.from_pretrained(
                torch.FloatTensor(embed_weights),
                freeze=freeze_embed)
        else:
            self.embedder = nn.Embedding(
                num_embeddings=6488 + 2,
                embedding_dim=embed_size,
                padding_idx=0)

        self.dropout = nn.Dropout(dropout_rate)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout_rate)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, inputs, targets=None):
        src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S
        embeds = self.forward_to_embeds(inputs)

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # S x N x E
        encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x S x E
        pooled_outputs = torch.mean(encoded_inputs, dim=1)

        return pooled_outputs

    def forward_to_embeds(self, inputs):
        inputs = inputs.to(torch.int64)
        embeds = self.embedder(inputs) * math.sqrt(self.embed_size)  # N x S x E
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # S x N x E
        embeds = embeds.to(self.device)

        return embeds


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class MLP(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(MLP, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        input_p2 = self.post_dropout(input_p1)
        output = self.post_layer_2(input_p2)
        return output


class FFNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.5, device="cpu"):
        super(FFNEncoder, self).__init__()

        num_middle_layers = num_layers - 2
        assert num_middle_layers >= 0, "Number of layers must be at least 2"

        self.dropout0 = nn.Dropout(p=dropout_prob)

        # Hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization after hidden layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_middle_layers):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(p=dropout_prob))

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x):
        x = x.to(self.device)

        x = self.dropout0(x)

        # first layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # middle layers
        for fc, bn, relu, dropout in zip(self.fcs, self.bns, self.relus, self.dropouts):
            x = fc(x)
            x = bn(x)
            x = relu(x)
            x = dropout(x)

        # Output layer
        x = self.fc2(x)

        return x

