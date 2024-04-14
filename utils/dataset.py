from torch.utils.data import Dataset


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
