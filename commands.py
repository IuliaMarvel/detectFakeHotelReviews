import fire


from main_scripts.train import main as main_train
from main_scripts.infer import main as main_infer


if __name__ == "__main__":
    fire.Fire(
        {
            "infer": main_infer,
            "train": main_train,
        }
    )
