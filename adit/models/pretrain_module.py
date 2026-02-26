import os
from typing import Any, Dict, Tuple, Optional
import torch
from lightning import LightningModule
from torchmetrics import MinMetric

from adit.models.task.noiser import StructureNoiser
from adit.models.loss import DenoisingLoss


class BaseLitModule(LightningModule):
    """Example of a `LightningModule` for any type of training.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_monitor: str,
        compile: bool,
    ) -> None:
        """Initialize a `LightningModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        # network in torch
        self.net = net
        
        # for averaging loss across batches
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`. 
        (Not actually used)

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        self.val_loss_best.reset()
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], training: Optional[bool] = True
    ):
        raise NotImplementedError("Model step not implemented.")
    
    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ):
        raise NotImplementedError("Predict step not implemented.")
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_bd = self.model_step(batch)

        # update and log metrics
        # self.train_loss(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for k,v in loss_bd.items():
            if k == 'loss': continue
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_bd = self.model_step(batch, training=False)

        # update and log metrics
        # self.val_loss(loss) # update
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # _vall = self.val_loss.compute()  # get current val acc
        # self.val_loss_best(_vall)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        return 

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError("Test step not implemented.")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=[p for p in self.trainer.model.parameters() if p.requires_grad])
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class DenoisingModule(BaseLitModule):
    """Structure Denoising for Pre-Training.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_monitor: str,
        noise_model: StructureNoiser,
        loss: Dict[str, Any],
        compile: bool,
        loss_type: str,
        output_param: str,
        save_dir = "./repr"
    ) -> None:
        """Initialize a `ResidueTypePredictionModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(net, optimizer, scheduler, scheduler_monitor, compile)
        # network and diffusion module
        self.noise_model = noise_model
        # loss function
        self.loss = DenoisingLoss(config=self.hparams.loss)
        self.loss_type = loss_type
        self.output_param = output_param
        assert output_param in ["v", "x_1"]

        self.layernorm_atom = torch.nn.LayerNorm(net.atom_dim)
        self.linear_no_bias_position_update = torch.nn.Linear(net.atom_dim, 3, bias=False)
        self.save_dir = os.path.expanduser(save_dir)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of losses break-down.
        """
        batch = self.noise_model.add_noise(batch, self.loss_type)

        # feedforward
        out = self.net(batch)
        position_update = self.linear_no_bias_position_update(
            self.layernorm_atom(out["atom_feat"])
        )
        atom_mask = batch["atom_mask"].bool()
        xyz = batch["atom_positions"][atom_mask]
        if self.output_param == "v":
            out["pred_atom_positions"] = xyz + position_update
        else:
            out["pred_atom_positions"] = position_update
        
        # calculate losses
        loss, loss_bd = self.loss(out, batch, _return_breakdown=True)
        return loss, loss_bd
