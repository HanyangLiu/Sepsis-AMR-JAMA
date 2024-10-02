import torch
from torch import optim
from models._base_mm import BaseMM
from utils.types_ import *
import lightning as L
import numpy as np
from sklearn import metrics


class experiment(L.LightningModule):
    def __init__(self,
                 base_model: BaseMM,
                 params: dict,
                 all_params: dict) -> None:
        super(experiment, self).__init__()
        self.save_hyperparameters(all_params)

        self.model = base_model
        self.params = params
        self.all_params = all_params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        self.eval_step_y_true = []
        self.eval_step_y_prob = []

    def forward(self, input, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        self.curr_device = batch["label"].device

        results = self.forward(batch, current_epoch=self.current_epoch)
        train_loss = self.model.loss_function(results,
                                              targets=batch["label"],
                                              batch_idx=batch_idx,
                                              epoch_idx=self.current_epoch)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, prog_bar=True)

        return train_loss['loss']

    def on_train_start(self):
        params = self.all_params['model_params']
        params.update(self.all_params['data_params'])
        params.update(self.all_params['exp_params'])
        self.logger.log_hyperparams(params=params)

    def _log_multiclass(self, AUROCs, AUPRCs):
        self.log('SS_auroc', AUROCs[0], prog_bar=True)
        self.log('SS_auprc', AUPRCs[0], prog_bar=True)
        self.log('RS_auroc', AUROCs[1], prog_bar=True)
        self.log('RS_auprc', AUPRCs[1], prog_bar=True)
        self.log('RR_auroc', AUROCs[2], prog_bar=True)
        self.log('RR_auprc', AUPRCs[2], prog_bar=True)

    def _eval_auc(self, y_true, y_prob):
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        prec, rec, _ = metrics.precision_recall_curve(y_true, y_prob)
        auroc = metrics.auc(fpr, tpr)
        auprc = metrics.auc(rec, prec)
        return auroc, auprc

    def _multiclass_auc(self, y_true, y_prob):
        AUROCs, AUPRCs = [], []
        for label in range(y_prob.shape[1]):
            gt = (y_true == label).astype(int).reshape((-1,))
            prob = y_prob[:, label]
            auroc, auprc = self._eval_auc(gt, prob)
            AUROCs.append(auroc)
            AUPRCs.append(auprc)
        return AUROCs, AUPRCs

    def _eval_cosine(self, features):
        z_MM, z_UMs = features[0], features[1:]
        z_MM = z_MM / z_MM.norm(dim=-1, keepdim=True)
        sim = torch.tensor(0)
        for z in z_UMs:
            z = z / z.norm(dim=-1, keepdim=True)
            sim += torch.sum(z_MM * z, dim=-1).mean()
        return sim

    def _shared_eval_step(self, batch, batch_idx, prefix=None):
        self.curr_device = batch["label"].device

        results = self.forward(batch, current_epoch=self.current_epoch)
        val_loss = self.model.loss_function(results,
                                            targets=batch["label"],
                                            batch_idx=batch_idx,
                                            epoch_idx=self.current_epoch)

        out = self.model.predict_proba(batch, current_epoch=self.current_epoch)
        y_prob = torch.stack(out, dim=2) if type(out) == list else out

        self.eval_step_y_prob.append(y_prob)
        self.eval_step_y_true.append(batch["label"])

        self.log_dict({"{}".format(prefix) + f"_{key}": val.item() for key, val in val_loss.items()},
                      sync_dist=True,
                      prog_bar=True)

    def _shared_on_eval_epoch_end(self, prefix=None):
        y_true = torch.cat(self.eval_step_y_true, dim=0).cpu().numpy()
        y_probs = torch.cat(self.eval_step_y_prob, dim=0).cpu().numpy()
        if len(y_probs.shape) != 3:
            y_probs = np.reshape(y_probs, (-1, y_probs.shape[1], 1))

        for j in range(y_probs.shape[2]):
            y_prob = y_probs[:, :, j]

            # if multiclass
            n_classes = y_prob.shape[1]
            if n_classes > 1:
                if prefix == 'test':
                    AUROCs, AUPRCs = self._multiclass_auc(y_true, y_prob)
                    self._log_multiclass(AUROCs, AUPRCs)
                y_prob = 1 - y_prob[:, 0]
                y_true = (y_true > 0).astype(int).reshape((-1,))

            auroc, auprc = self._eval_auc(y_true, y_prob)
            self.eval_step_y_true.clear()
            self.eval_step_y_prob.clear()
            if j == 0:
                self.log('{}_auroc'.format(prefix), auroc, prog_bar=True)
                self.log('{}_auprc'.format(prefix), auprc, prog_bar=True)
                if prefix == 'val':
                    break
            else:
                self.log('{}_auroc_mod{}'.format(prefix, j), auroc, prog_bar=True)
                self.log('{}_auprc_mod{}'.format(prefix, j), auprc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, prefix='val')

    def on_validation_epoch_end(self) -> None:
        self._shared_on_eval_epoch_end(prefix='val')

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, prefix='test')

    def on_test_epoch_end(self) -> None:
        self._shared_on_eval_epoch_end(prefix='test')

    def predict_step(self, batch, batch_idx) -> dict:
        labels = batch["label"]
        y_prob = self.model.predict_proba(batch, current_epoch=self.current_epoch)
        if isinstance(y_prob, List):
            y_prob = torch.stack(y_prob, dim=1)  # [B, M, 1]

        features = self.model.encode(batch, current_epoch=self.current_epoch)
        features = [feat / feat.norm(dim=-1, keepdim=True) for feat in features]
        features = torch.stack(features, dim=1)  # [B, M, D]

        try:
            evid_mm, evid_um, _, _, uncertainties = self.forward(batch, current_epoch=self.current_epoch)
            E = torch.stack([evid_mm] + evid_um, dim=2)
            U = torch.stack([uncertainties[v] for v in range(4)], dim=1)  # [B, M, D]
            return {
                'features': features,
                'probs': y_prob,
                'labels': labels,
                'evidence': E,
                'uncertainty': U,
            }
        except:
            return {
                'features': features,
                'probs': y_prob,
                'labels': labels
            }

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
