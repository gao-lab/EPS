import logging

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


def build_mlp(dims: list[int], dropout: float = 0.1) -> nn.Module:
    mlp = []
    for i in range(0, len(dims) - 2):
        mlp.append(nn.Linear(dims[i], dims[i + 1], bias=True))
        mlp.append(nn.BatchNorm1d(dims[i + 1]))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(dropout))
    mlp.append(nn.Linear(dims[-2], dims[-1], bias=True))
    return nn.Sequential(*mlp)


class MLPRegressModel(LightningModule):
    r"""
    Baseline model for training and evaluation.

    Args:
        dims: list of layer dimensions, including input and output dimensions
        dropout: dropout rate, default 0.0
        lr: learning rate, default 1e-3
        weight_decay: weight decay, default 1e-5
    """

    def __init__(
        self, dims: list[int], dropout: float = 0.0, lr: float = 1e-3, weight_decay: float = 1e-5
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_mlp(dims, dropout=dropout)
        self.loss = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_list = []
        self.apply(self._init_weights_module)  # init weight

    def _init_weights_module(self, module, std=0.02, trunc=2) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0, std=std, a=-trunc, b=trunc)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self.backbone(batch[0])
        self.test_list.append(y_hat.detach().cpu().float().numpy())

    def on_test_end(self) -> np.ndarray:
        self.pred = np.concatenate(self.test_list, axis=0)
        self.test_list = []


class TorchMLPRegressor:
    r"""
    MLP Regressor via PyTorch

    Args:
        seed: random seed
    """

    def __init__(self, seed: int = 0):
        self.init_trainer()
        logging.getLogger("pytorch_lightning").setLevel(logging.FATAL)
        # set seed
        import torch  # local import

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        dims = [x.shape[-1]] + [128] + [y.shape[-1]]
        self.model = MLPRegressModel(dims=dims, dropout=0.1)
        train_dataset = TensorDataset(Tensor(x.copy()), Tensor(y.copy()))
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2
        )
        self.trainer.fit(self.model, train_loader)

    def predict(self, x: np.ndarray) -> np.ndarray:
        test_dataset = TensorDataset(Tensor(x.copy()))
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)
        self.trainer.test(self.model, test_loader)
        return self.model.pred

    def init_trainer(self, epochs: int = 100, metric: str = "train/total_loss", patient: int = 10):
        r"""
        Init pytorch lightning trainer
        """
        torch.set_float32_matmul_precision("high")
        # disable slurm (only needed in ray)
        env = LightningEnvironment()
        plugins = [env]

        # callbacks
        early_stop = EarlyStopping(monitor=metric, patience=patient, mode="min")
        callback_list = [early_stop]

        # init trainer
        self.trainer = Trainer(
            accelerator="auto",
            callbacks=callback_list,
            devices=1,
            precision="16-mixed",
            logger=False,
            log_every_n_steps=1,
            max_epochs=epochs,
            plugins=plugins,
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
