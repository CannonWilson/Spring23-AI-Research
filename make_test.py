import os
import sys
from dotenv import load_dotenv
import shutil
from itertools import islice
from pathlib import Path
import pandas as pd

load_dotenv()

print("Making test dir")

SHOULD_COPY = True # Should the img files be copied or just moved?

CELEB_DIR = os.getenv("CELEBA_DIR")
assert os.path.exists(CELEB_DIR), \
    "CelebA directory was not successfully installed"
PARTITION_FILE = 'list_eval_partition.txt'
assert os.path.exists(PARTITION_FILE), \
    "list_eval_partition.txt is not in root directory"
CSV_FILE = 'list_attr_celeba.csv'
assert os.path.exists(CSV_FILE), \
    "list_attr_celeba.csv is not in root directory"

TRAIN_DIR, VAL_DIR, TEST_DIR = os.getenv("TRAIN_DIR"), os.getenv("VAL_DIR"), os.getenv("TEST_DIR")

SUB_DIRS = [os.path.join("old", "male", "smile"),
            os.path.join("old", "male", "no_smile"),
            os.path.join("old", "female", "smile"),
            os.path.join("old", "female", "no_smile"),
            os.path.join("young", "male", "smile"),
            os.path.join("young", "male", "no_smile"),
            os.path.join("young", "female", "smile"),
            os.path.join("young", "female", "no_smile")]

for sdir in SUB_DIRS:
    te_dir = os.path.join(TEST_DIR, sdir)
    Path(te_dir).mkdir(parents=True, exist_ok=True)

train_path = Path(TRAIN_DIR)
val_path = Path(VAL_DIR)
train_paths = [i.path for i in islice(os.scandir(train_path), None)]
val_paths = [i.path for i in islice(os.scandir(val_path), None)]

celeb_dir_path = Path(CELEB_DIR)
celeb_paths = [i.path for i in islice(os.scandir(celeb_dir_path), None)]
celeba_df = pd.read_csv(CSV_FILE)

with open(PARTITION_FILE, encoding='utf-8') as f:
    for line in f:
        f_name, partition = line.split()
        f_path = os.path.join(CELEB_DIR, f_name)
        if partition == "2":
            if f_path not in train_paths and f_path not in val_paths:
                view = celeba_df[celeba_df['filename'] == f_name]
                age = "young" if view['Young'].item() == 1 else "old" #pylint: disable=invalid-name
                sex = "male" if view['Male'].item() == 1 else "female" #pylint: disable=invalid-name
                smile = "smile" if view['Smiling'].item() == 1 else "no_smile" #pylint: disable=invalid-name
                destination = os.path.join(TEST_DIR, age, sex, smile)
                if SHOULD_COPY:
                    shutil.copy(f_path, destination)
                else: # if not copying, just move file
                    shutil.move(f_path, destination)

print("Finished making test set.")
