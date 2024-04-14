import os
import shutil


def detect_polarity(data_dir):
    folders = data_dir.split("/")
    for folder in folders:
        if "polarity" in folder:
            polarity = folder.split("_")[0]

    return polarity


def merge_folders(source_dir, destination_dir):
    polarity = detect_polarity(source_dir)

    src_folders = os.listdir(source_dir)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for src_folder in src_folders:
        files = os.listdir(os.path.join(source_dir, src_folder))

        for file in files:
            src_path = os.path.join(os.path.join(source_dir, src_folder), file)
            dst_path = os.path.join(destination_dir, polarity + "_" + file)
            shutil.copyfile(src_path, dst_path)


files_dirs = [
    "new_data/op_spam_v1.4/negative_polarity/truthful_from_Web",
    "new_data/op_spam_v1.4/negative_polarity/deceptive_from_MTurk",
    "new_data/op_spam_v1.4/positive_polarity/deceptive_from_MTurk",
    "new_data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor",
]

destination_dir = "new_data/merged"

for files_dir in files_dirs:
    merge_folders(files_dir, destination_dir)

print("Data folder has been formatted properly.")
