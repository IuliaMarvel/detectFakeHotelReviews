import os

import torch
from tqdm import tqdm

from main_scripts.train import DeceptiveReviewClassifier


def load_model(model_path):
    model = DeceptiveReviewClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_review_deceptiveness(model, review):
    with torch.no_grad():
        logits = model(review)
        probabilities = torch.softmax(logits, dim=1)
        prob_deceptive = probabilities[:, 1].item()

    return prob_deceptive


def main(model_path, test_dir, output_file):
    model = load_model(model_path)

    reviews = os.listdir(test_dir)

    predictions = []

    for review in tqdm(reviews):
        with open(test_dir + review, "r") as f:
            review_text = f.read()

        probability = predict_review_deceptiveness(model, [review_text])
        predictions.append(str(probability))

    with open(output_file, "w") as f:
        f.write("\n".join(predictions))
