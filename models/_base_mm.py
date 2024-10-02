import os.path

import torch
from torch import nn
from utils.types_ import *
import numpy as np
import pandas as pd
from ._modules import Transformer, FFNEncoder
from ._grud import BackboneGRUD
from paths import processed_data_path


class BaseMM(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 embed_size: int,
                 embed_weights: str,
                 max_codes: int,
                 static_hidden: int = 32,
                 use_pretrain: bool = False,
                 freeze_embed: bool = False,
                 num_layers: int = 1,
                 num_heads: int = 2,
                 dropout: float = 0.3,
                 multiclass: bool = False,
                 clf_thresh: float = 0.5,
                 **kwargs) -> None:
        super(BaseMM, self).__init__()

        self.compute_device = kwargs['device']
        self.multiclass = multiclass
        self.out_dim = 3 if self.multiclass else 1
        self.clf_thresh = clf_thresh
        self.tau = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=False)

        df_embedding = pd.read_csv(os.path.join(processed_data_path, embed_weights)).drop(columns=['code'])
        weights = np.zeros((len(df_embedding) + 2, embed_size))
        weights[1:-1, :] = df_embedding.values  # "0" as padding, largest idx to mark all empty

        self.encoder_S = FFNEncoder(
            input_dim=kwargs["size_static"][0],
            hidden_dim=static_hidden,
            output_dim=embed_size,
            num_layers=2,
            dropout_prob=dropout,
            device=self.compute_device
        )
        self.encoder_T = BackboneGRUD(
            n_steps=kwargs["size_timeseries"][0],
            n_features=kwargs["size_timeseries"][1],
            rnn_hidden_size=embed_size,
        )
        self.encoder_C = Transformer(
            weights, embed_size, max_codes, use_pretrain, freeze_embed, num_layers,
            num_heads=num_heads,
            output_size=embed_size,
            dropout_rate=dropout,
            device=self.compute_device
        )
        self.fc_mm = nn.Linear(embed_size * 3, embed_size)

        # binary output
        self.sigmoid = nn.Sigmoid()
        self.cls_loss_binary = nn.BCEWithLogitsLoss(reduction='mean')

        # multiclass output
        self.softmax = nn.Softmax(dim=-1)
        self.cls_loss_multi = nn.CrossEntropyLoss(reduction='mean')

    def logits_to_probs(self, logits):
        return self.softmax(logits) if self.multiclass else self.sigmoid(logits)

    def prediction_loss(self, logits, targets):
        return self.cls_loss_multi(logits, targets.view((-1,)).long()) \
            if self.multiclass else self.cls_loss_binary(logits, targets.float())

    def predict_proba(self, input, **kwargs):
        pass

    def forward(self, batch, **kwargs) -> Tensor:
        pass

    def loss_function(self, *args, **kwargs) -> dict:
        pass

    def encode(self, input, **kwargs):
        pass

    def _eval_cosine(self, features):
        z_mm, z_um = features[0], features[1:]
        z_mm = z_mm / z_mm.norm(dim=-1, keepdim=True)
        sim = torch.tensor(0.0).to(self.compute_device)
        for z in z_um:
            z = z / z.norm(dim=-1, keepdim=True)
            sim += torch.sum(z_mm * z, dim=-1).mean()
        return sim / len(z_um)