import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl


class HotelReviewsDataset(Dataset):
    def __init__(self, reviews):
        self.reviews = reviews
        self.labels = [0 if "_t_" in f else 1 for f in self.reviews]

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        with open(self.reviews[idx], "r") as f:
            review = f.read()

        label = self.labels[idx]

        return {"text": review, "label": label}


class HotelReviewsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, val_split=0.5, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state

    def setup(self, stage=None):
        reviews = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if not f.startswith(".")]
        
        reviews_train, reviews_val = train_test_split(reviews, test_size=self.val_split, random_state=self.random_state)
        
        self.train_dataset = HotelReviewsDataset(reviews_train)
        self.val_dataset = HotelReviewsDataset(reviews_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)