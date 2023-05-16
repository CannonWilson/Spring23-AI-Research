"""
PLEASE BE ADVISED: 
This scipt will delete any 
existing train/val/test directories.
                
Fill all of the training and 
validation sub-directories with 
files from the CelebA directory.
The test set is filled with files 
from the original CelebA test 
split that aren't already 
included in the training/val sets.
"""

import os
import shutil
from itertools import islice
from pathlib import Path
import pandas as pd
from settings import *

SHOULD_COPY = True # Should the img files be copied or just moved?
copy_or_move = shutil.copy if SHOULD_COPY else shutil.move

# Check all required dirs/files are installed
assert os.path.exists(CELEBA_DIR), \
    f"Expected CelebA directory installed at path {CELEBA_DIR}"
assert os.path.exists(CELEBA_PART_TXT), \
    f"Expected partition text file at path {CELEBA_PART_TXT}"
assert os.path.exists(CELEBA_ATTRS_CSV), \
    f"Expected attributes csv file at path {CELEBA_ATTRS_CSV}"

# Remove train/val/test directories if they already exist
for data_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        print("Found existing TRAIN|VAL|TEST dir. Removing.")
        shutil.rmtree(data_dir)

SUB_DIRS = SUBDIRS_1_CORR if NUM_CORRS == 1 else SUBDIRS_2_CORR
for sdir in SUB_DIRS:
    Path(os.path.join(TRAIN_DIR, sdir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(VAL_DIR, sdir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(TEST_DIR, sdir)).mkdir(parents=True, exist_ok=True)

celeb_dir_path = Path(CELEBA_DIR)
celeb_paths = [i.path for i in islice(os.scandir(celeb_dir_path), None)]
celeba_df = pd.read_csv(CELEBA_ATTRS_CSV)

# (NEWEST) variables for dataset with 2 correlations - male, smile
# 4:1 correlation for sex, 2:1 correlation for smiling
TRAIN_LIMS = TRAIN_LIMS_1_CORR if NUM_CORRS == 1 else TRAIN_LIMS_2_CORR
train_counts = {subgroup: 0 for subgroup in TRAIN_LIMS}
VAL_LIMS = VAL_LIMS_1_CORR if NUM_CORRS == 1 else VAL_LIMS_2_CORR
val_counts = {subgroup: 0 for subgroup in VAL_LIMS}

# Read in attributes from csv
celeba_df = pd.read_csv(CELEBA_ATTRS_CSV)

print("Filling TRAIN and VAL directories")

for f_idx, f_path in enumerate(celeb_paths):

    if f_idx%10000 == 0:
        print('Now checking file number ', f_idx)

    f_name = f_path.split('/')[-1]
    view = celeba_df[celeba_df['filename'] == f_name]
    age = "young" if view['Young'].item() == 1 else "old"
    sex = "male" if view['Male'].item() == 1 else "female"
    smile = "smile" if view['Smiling'].item() == 1 else "no_smile"
    full_key = "-".join((age, sex)) if NUM_CORRS == 1 else \
        "-".join((age, sex, smile))

    if train_counts[full_key] < TRAIN_LIMS[full_key]: # add file to training data
        destination_path = "/".join([TRAIN_DIR, *full_key.split("-")])
        copy_or_move(f_path, destination_path)
        train_counts[full_key] = train_counts[full_key] + 1

    elif val_counts[full_key] < VAL_LIMS[full_key]: # add file to validation data
        destination_path = "/".join([TRAIN_DIR, *full_key.split("-")])
        copy_or_move(f_path, destination_path)
        val_counts[full_key] = val_counts[full_key] + 1

print("Finished creating training and validation sets.")
print("TRAIN COUNTS: ", train_counts)
print("VAL COUNTS: ", val_counts)

# Go through the eval partitions
# file and move the file into the
# test folder if possible
train_path = Path(TRAIN_DIR)
val_path = Path(VAL_DIR)
train_paths = [i.path for i in islice(os.scandir(train_path), None)]
val_paths = [i.path for i in islice(os.scandir(val_path), None)]

with open(CELEBA_PART_TXT, encoding='utf-8') as f:
    for line in f:
        f_name, partition = line.split()
        f_path = os.path.join(CELEBA_DIR, f_name)
        if partition == "2": # "0" is train, "1" is val, "2" is test
            if f_path in celeb_paths and \
                    f_path not in train_paths and \
                    f_path not in val_paths:
                view = celeba_df[celeba_df['filename'] == f_name]
                age = "young" if view['Young'].item() == 1 else "old"
                sex = "male" if view['Male'].item() == 1 else "female"
                smile = "smile" if view['Smiling'].item() == 1 else "no_smile"
                destination = os.path.join(TEST_DIR, age, sex) if NUM_CORRS == 1 \
                    else os.path.join(TEST_DIR, age, sex, smile)
                copy_or_move(f_path, destination_path)

print("Finished making test set.")
