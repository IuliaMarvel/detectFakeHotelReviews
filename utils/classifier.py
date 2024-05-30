import mlflow
import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer


class DeceptiveReviewClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=2e-5):
        super(DeceptiveReviewClassifier, self).__init__()
        self.labse = SentenceTransformer("sentence-transformers/LaBSE")
        self.labse.requires_grad_(False)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, num_classes),
        )
        self.lr = lr

    def forward(self, reviews):
        embeddings = torch.Tensor(self.labse.encode(reviews))
        logits = self.classifier(embeddings)
        return logits

    def training_step(self, batch):
        reviews = batch["text"]
        labels = batch["label"]
        logits = self.forward(reviews)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        # self.log("train_loss", loss)
        mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        reviews = batch["text"]
        labels = batch["label"]
        logits = self.forward(reviews)
        logits = self.forward(reviews)
        val_loss = torch.nn.functional.cross_entropy(logits, labels)
        # self.log("val_loss", val_loss)
        mlflow.log_metric("val_loss", val_loss.item(), step=self.global_step)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)
        return optimizer
