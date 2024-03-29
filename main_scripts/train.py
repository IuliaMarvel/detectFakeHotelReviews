import torch
import os
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


class HotelReviewsDataset(Dataset):
    def __init__(self, reviews):

        self.reviews = reviews
        self.labels = [0 if '_t_' in f else 1 for f in self.reviews]

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):

        with open(self.reviews[idx], 'r') as f:
            review = f.read()

        label = self.labels[idx]

        return {'text': review, 'label': label}
    
class DeceptiveReviewClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=2e-5):
        super(DeceptiveReviewClassifier, self).__init__()
        self.labse = SentenceTransformer('sentence-transformers/LaBSE')
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
        reviews = batch['text']
        labels = batch['label']
        logits = self.forward(reviews)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
            reviews = batch['text']
            labels = batch['label']
            logits = self.forward(reviews)
            logits = self.forward(reviews)
            val_loss = torch.nn.functional.cross_entropy(logits, labels)
            self.log('val_loss', val_loss)
            return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)
        return optimizer

def train_model(train_loader, val_loader, num_classes=2, max_epochs=1, lr=2e-5):
    model = DeceptiveReviewClassifier(num_classes=num_classes, lr=lr)
    accelerator = "cuda" if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator)

    trainer.fit(model, train_loader, val_loader)

    return model

def main(data_dir, batch_size, epochs):

    data_dir = data_dir #'data/merged/'
    reviews = [data_dir + f for f in os.listdir(data_dir) if not(f.startswith('.'))][:20]

    reviews_train, reviews_val = train_test_split(reviews, test_size=0.5, random_state=42)

    train_dataset = HotelReviewsDataset(reviews_train)
    val_dataset = HotelReviewsDataset(reviews_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = train_model(train_loader, val_loader, max_epochs=epochs)
    torch.save(model.state_dict(), 'deceptive_review_classifier.pt')


