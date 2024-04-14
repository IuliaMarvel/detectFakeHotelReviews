import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.classifier import DeceptiveReviewClassifier
from utils.dataset import HotelReviewsDataset


def train_model(train_loader, val_loader, num_classes=2, max_epochs=1, lr=2e-5):
    model = DeceptiveReviewClassifier(num_classes=num_classes, lr=lr)
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator)

    trainer.fit(model, train_loader, val_loader)

    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    data_dir = cfg.train.data_dir
    reviews = [data_dir + f for f in os.listdir(data_dir) if not (f.startswith("."))][
        :20
    ]

    reviews_train, reviews_val = train_test_split(
        reviews, test_size=0.5, random_state=42
    )

    train_dataset = HotelReviewsDataset(reviews_train)
    val_dataset = HotelReviewsDataset(reviews_val)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    model = train_model(train_loader, val_loader, max_epochs=cfg.train.epochs)
    torch.save(model.state_dict(), "deceptive_review_classifier.pt")


if __name__ == "__main__":
    main()
