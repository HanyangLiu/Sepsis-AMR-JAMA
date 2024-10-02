import torch
from utils.types_ import *
from ._modules import MLP
from ._base_mm import BaseMM


class DeepModel(BaseMM):
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
                 **kwargs) -> None:
        super(DeepModel, self).__init__(
            embed_size,
            embed_weights,
            max_codes,
            static_hidden,
            use_pretrain,
            freeze_embed,
            num_layers,
            num_heads,
            dropout,
            multiclass,
            **kwargs
        )
        self.predictor = MLP(dropout=dropout,
                             in_dim=embed_size,
                             post_dim=embed_size,
                             out_dim=self.out_dim)

    def predict_proba(self, batch, **kwargs):
        logits, _, _ = self.forward(batch)
        probs = self.softmax(logits) if self.multiclass else self.sigmoid(logits)
        return probs

    def forward(self, batch, **kwargs) -> Tensor:
        data_S, data_T, data_C, mod_mask = batch["static"], \
                                           batch["ts"], \
                                           batch["comorb"], \
                                           batch["modality_mask"]

        # unimodal embeddings
        z_S = self.encoder_S(data_S)
        _, z_T = self.encoder_T(
            data_T["X"],
            data_T["mask"],
            data_T["delta"],
            data_T["mean"],
            data_T["X_LOCF"],
        )
        z_C = self.encoder_C(data_C)

        # fusion
        embed = torch.concat([z_S, z_T, z_C], dim=-1)
        z_mm = self.fc_mm(embed)
        logits = self.predictor(z_mm)

        return logits, z_mm, [z_S, z_T, z_C]

    def loss_function(self, *args, **kwargs) -> dict:
        logits, z_mm, z_um = args[0]
        targets = kwargs['targets']
        loss = self.cls_loss_multi(logits, targets.view((-1,)).long()) \
            if self.multiclass else self.cls_loss_binary(logits, targets.float())
        mm_similarity = self._eval_cosine([z_mm] + z_um)
        uni_similarity = self._eval_cosine(z_um)

        return {
            "loss": loss,
            "mm_similarity": mm_similarity,
            "um_similarity": uni_similarity
        }

    def encode(self, input, **kwargs):
        _, z_mm, z_um = self.forward(input, **kwargs)
        return [z_mm] + z_um
