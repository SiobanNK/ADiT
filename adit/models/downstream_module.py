import os
from typing import Any, Dict, Tuple, Optional
import torch
import pickle as pkl
from torcheval.metrics.functional import r2_score
from lifelines.utils import concordance_index

def reorder_args_concordance_index(pred, target):
    target = target.cpu()
    pred = pred.cpu()
    return concordance_index(target, pred)

from adit.common.variadic_utils import spearmanr, pearsonr, f1_max
from adit.models.pretrain_module import BaseLitModule


class Metrics:
    """
    union of all metrics used in downstream tasks
    including mse_loss, mae_loss, spearmanr, pearsonr, r2_score
    inputs: **(pred, target)
    """
    mapping = {
        'mse_loss': torch.nn.functional.mse_loss,
        'mae_loss': torch.nn.functional.l1_loss,
        'bce_loss': torch.nn.BCELoss(),
        'bce_loss_logit': torch.nn.BCEWithLogitsLoss(),
        'spearmanr': spearmanr,
        'pearsonr': pearsonr,
        'r2_score': r2_score,
        'concordance_index': reorder_args_concordance_index,
        'f1_max': f1_max
    }

    def __init__(self, metrics: list[str], metrics_test: list[str] = None):
        self.metrics = metrics
        self.funcs = [self.mapping[name] for name in metrics]

        self.metrics_test = metrics_test
        if self.metrics_test is not None:
            self.funcs_test = [self.mapping[name] for name in metrics_test]

    def compute(self, pred: torch.tensor, target: torch.tensor, mode:str = 'valid'):
        if 'f1_max' not in self.metrics:
            pred = pred.view(-1)
            target = target.view(-1)
        assert pred.shape[0] == target.shape[0]
        assert mode in ["valid", "val", "test"]

        if mode == "test" and self.metrics_test is not None:
            result = {
                self.metrics_test[i]: self.funcs_test[i](pred, target)
                for i in range(len(self.metrics_test))
            }
        else:
            result = {
                self.metrics[i]: self.funcs[i](pred, target)
                for i in range(len(self.metrics))
            }
        return result


class BaseDownstreamModule(BaseLitModule):

    def __init__(
        self,
        net: torch.nn.Module,
        mlp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_monitor: str,
        loss: Dict[str, Any],
        metrics: Metrics,
        compile: bool,
        save_file = None
    ) -> None:
        super().__init__(net, optimizer, scheduler, scheduler_monitor, compile)
        self.mlp = mlp
        self.loss = loss
        self.metrics = metrics
        self.save_file = save_file # saving detailed test result: (all_pred, all_target)

        # cache results
        self.valid_step_preds = []
        self.valid_step_targets = []
        self.test_step_preds = []
        self.test_step_targets = []
        self.test_accession_code = []

    def forward(self, batch):
        batch = self.net(batch)
        pred = self.mlp(batch['complex_feat']).view(-1)
        return pred
    
    def get_target(self, batch):
        raise NotImplementedError
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.forward(batch)
        target = self.get_target(batch)
        cum_loss, losses = self.loss(pred, target)

        if training:
            return cum_loss, losses
        else:
            return cum_loss, losses, pred, target

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, loss_bd, pred, target = self.model_step(batch, training=False)
        self.valid_step_preds.append(pred)
        self.valid_step_targets.append(target)

    def on_validation_epoch_end(self) -> None:
        all_preds = torch.concat(self.valid_step_preds, dim=-1)
        all_targets = torch.concat(self.valid_step_targets, dim=-1)

        results = self.metrics.compute(all_preds, all_targets)
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)

        self.valid_step_preds.clear()
        self.valid_step_targets.clear()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_bd, pred, target = self.model_step(batch, training=False)
        self.test_step_preds.append(pred)
        self.test_step_targets.append(target)
        if "accession_code" in batch.keys():
            self.test_accession_code = self.test_accession_code + batch["accession_code"]

    def on_test_epoch_end(self) -> None:
        all_preds = torch.cat(self.test_step_preds, dim = -1)
        all_targets = torch.cat(self.test_step_targets, dim = -1)

        results = self.metrics.compute(all_preds, all_targets, mode='test')
        for k, v in results.items():
            self.log(f"test/{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if self.save_file is not None:
            with open(self.save_file, "wb") as f:
                pkl.dump(
                    (
                        self.test_accession_code, all_preds, all_targets
                    ), f, protocol=pkl.HIGHEST_PROTOCOL
                )

        self.test_step_preds.clear()
        self.test_step_targets.clear()
        self.test_accession_code.clear()


class SkempiModule(BaseDownstreamModule):
    """Skempi dataset, pridict ddG
    """

    def forward(self, batch):
        wt_batch, mt_batch, batch_ddG = self.data_object_split(batch)
        wt_prot_feat = self.model_step_one_protein(wt_batch)
        mt_prot_feat = self.model_step_one_protein(mt_batch)

        outputs = torch.cat([mt_prot_feat, wt_prot_feat], dim=-1)
        pred = self.mlp(outputs).view(-1)
        outputs = torch.cat([wt_prot_feat, mt_prot_feat], dim=-1)
        pred = pred - self.mlp(outputs).view(-1)
        
        return pred
    
    def model_step_one_protein(self, batch):
        # feedforward
        batch = self.net(batch)
        prot_feat = batch['complex_feat']
        
        return prot_feat

    def data_object_split(self, batch):
        batch_ddG = batch["ddG"]
        mt_batch = {
            k[3:]: batch[k]
            for k in batch.keys() if k.startswith("mt_")
        }
        wt_batch = {
            k[3:]: batch[k]
            for k in batch.keys() if k.startswith("wt_")
        }

        return wt_batch, mt_batch, batch_ddG

    def get_target(self, batch):
        wt_batch, mt_batch, batch_ddG = self.data_object_split(batch)
        return batch_ddG.view(-1)


class LBAModule(BaseDownstreamModule):
    """LBA dataset, pridict neglog_aff
    """

    def get_target(self, batch):
        target = batch["neglog_aff"].view(-1)
        return target


class DavisModule(BaseDownstreamModule):
    """Davis dataset
    """

    def get_target(self, batch):
        target = batch["target"].view(-1)
        return target
