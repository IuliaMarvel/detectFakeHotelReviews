import os

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.classifier import DeceptiveReviewClassifier
from utils.dataset import HotelReviewsDataModule


def train_model(data_module, num_classes, max_epochs, lr):
    model = DeceptiveReviewClassifier(num_classes=num_classes, lr=lr)
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator)

    trainer.fit(model, data_module)

    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    data_module = HotelReviewsDataModule(
        data_dir=cfg.train.data_dir,
        batch_size=cfg.train.batch_size,
        val_split=cfg.val_split,
        random_state=cfg.random_state,
    )

    mlflow.set_tracking_uri(uri="http://mlflow_server:5000")
    mlflow.set_experiment("/reviews-check-experiment")
    with mlflow.start_run():
        model = train_model(
            data_module,
            num_classes=cfg.num_classes,
            max_epochs=cfg.train.epochs,
            lr=cfg.lr,
        )

    torch.save(model.state_dict(), cfg.infer.model_path)


if __name__ == "__main__":
    main()
