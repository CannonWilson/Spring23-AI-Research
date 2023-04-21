"""
Fill all of the training and 
validation sub-directories with 
files from the CelebA directory.
The test set is filled with files 
from the original CelebA test 
split that aren't already 
included in the training/val sets.
"""

import os
import sys
from dotenv import load_dotenv
import shutil
from itertools import islice
from pathlib import Path
import pandas as pd

load_dotenv()

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

# Create the train/val/test directories
TRAIN_DIR, VAL_DIR, TEST_DIR = os.getenv("TRAIN_DIR"), os.getenv("VAL_DIR"), os.getenv("TEST_DIR")

# response = input("The scipt will delete any "+\
#                  "existing train/val/test directory. "+\
#                  "Would you like to continue? yes/no: ")
# if response != "yes":
#     print("Did not type 'yes'. Exiting.")
#     sys.exit()

# # Remove train/val/test directories if they already exist
# for data_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
#     if os.path.exists(data_dir) and os.path.isdir(data_dir):
#         print("Found existing TRAIN|VAL|TEST dir. Removing.")
#         shutil.rmtree(data_dir)

# SUB_DIRS = [os.path.join("old", "male"),
#             os.path.join("old", "female"),
#             os.path.join("young", "male"),
#             os.path.join("young", "female")]

# for sdir in SUB_DIRS:
#     tr_dir = os.path.join(TRAIN_DIR, sdir)
#     va_dir = os.path.join(VAL_DIR, sdir)
#     te_dir = os.path.join(TEST_DIR, sdir)
#     Path(tr_dir).mkdir(parents=True, exist_ok=True)
#     Path(va_dir).mkdir(parents=True, exist_ok=True)
#     Path(te_dir).mkdir(parents=True, exist_ok=True)

celeb_dir_path = Path(CELEB_DIR)
celeb_paths = [i.path for i in islice(os.scandir(celeb_dir_path), None)]
celeba_df = pd.read_csv(CSV_FILE)

# # (ORIGINAL) variables for dataset with 1 correlation - sex w/ age
# TRAIN_LIMS = {
#     'old_male': 24812,
#     'old_female': 6203,
#     'young_male': 6203,
#     'young_female': 24812,
# }
# train_counts = {subgroup: 0 for subgroup in TRAIN_LIMS}
# VAL_LIMS = {
#     'old_male': 1000,
#     'old_female': 1000,
#     'young_male': 1000,
#     'young_female': 1000,
# }
# val_counts = {subgroup: 0 for subgroup in VAL_LIMS}

# # Read in attributes from csv
celeba_df = pd.read_csv(CSV_FILE)
# count = 0 # pylint: disable=invalid-name

# for f_path in celeb_paths:

#     if count%10000 == 0:
#         print('Now checking file number ', count)
#     count += 1

#     f_name = f_path.split('/')[-1]
#     view = celeba_df[celeba_df['filename'] == f_name]
#     age = "young" if view['Young'].item() == 1 else "old" # pylint: disable=invalid-name
#     sex = "male" if view['Male'].item() == 1 else "female" # pylint: disable=invalid-name
#     full_key = "_".join((age, sex)) # pylint: disable=invalid-name

#     if train_counts[full_key] < TRAIN_LIMS[full_key]:
#         # add file to training data
#         destination_path = os.path.join(TRAIN_DIR, age, sex)
#         if SHOULD_COPY:
#             shutil.copy(f_path, destination_path)
#         else: # if not copying, just move file
#             shutil.move(f_path, destination_path)
#         train_counts[full_key] = train_counts[full_key] + 1

#     elif val_counts[full_key] < VAL_LIMS[full_key]:
#         # add file to validation data
#         destination_path = os.path.join(VAL_DIR, age, sex)
#         if SHOULD_COPY:
#             shutil.copy(f_path, destination_path)
#         else: # if not copying, just move file
#             shutil.move(f_path, destination_path)
#         val_counts[full_key] = val_counts[full_key] + 1

# print("Finished creating training and validation sets.")
# print("TRAIN COUNTS: ", train_counts)
# print("VAL COUNTS: ", val_counts)

# Go through the eval partitions
# file and move the file into the
# test folder if possible
train_path = Path(TRAIN_DIR)
val_path = Path(VAL_DIR)
train_paths = [i.path for i in islice(os.scandir(train_path), None)]
val_paths = [i.path for i in islice(os.scandir(val_path), None)]


with open(PARTITION_FILE, encoding='utf-8') as f:
    for line in f:
        f_name, partition = line.split()
        f_path = os.path.join(CELEB_DIR, f_name)
        if partition == "2":
            if f_path in celeb_paths and \
                    f_path not in train_paths and \
                    f_path not in val_paths:
                view = celeba_df[celeba_df['filename'] == f_name]
                age = "young" if view['Young'].item() == 1 else "old" #pylint: disable=invalid-name
                sex = "male" if view['Male'].item() == 1 else "female" #pylint: disable=invalid-name
                destination = os.path.join(TEST_DIR, age, sex)
                if SHOULD_COPY:
                    shutil.copy(f_path, destination)
                else: # if not copying, just move file
                    shutil.move(f_path, destination)

print("Finished making test set.")
