import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from utils.classifier import DeceptiveReviewClassifier


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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    model = load_model(cfg.infer.model_path)

    reviews = os.listdir(cfg.infer.test_dir)

    predictions = []

    for review in tqdm(reviews):
        with open(cfg.infer.test_dir + review, "r") as f:
            review_text = f.read()

        probability = predict_review_deceptiveness(model, [review_text])
        predictions.append(str(probability))

    with open(cfg.infer.output_file, "w") as f:
        f.write("\n".join(predictions))


if __name__ == "__main__":
    main()
