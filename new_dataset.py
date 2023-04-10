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
import shutil
from itertools import islice
from pathlib import Path
import pandas as pd


CELEB_DIR = './img_align_celeba_3'
TRAIN_DIR = './train'
VAL_DIR = './val'
TEST_DIR = './test'
PARTITION_FILE = './list_eval_partition.txt'

celeb_dir_path = Path(CELEB_DIR)
celeb_paths = [i.path for i in islice(os.scandir(celeb_dir_path), None)]
celeba_df = pd.read_csv('list_attr_celeba.csv')

# (NEWEST) variables for dataset with 2 correlations - male, smile
# 4:1 correlation for sex, 2:1 correlation for smiling
TRAIN_LIMS = {
    'old_male_smile': 8000,
    'old_male_no_smile': 16000,
    'old_female_smile': 2000,
    'old_female_no_smile': 4000,
    'young_male_smile': 4000,
    'young_male_no_smile': 2000,
    'young_female_smile': 16000,
    'young_female_no_smile': 8000
}
train_counts = {subgroup: 0 for subgroup in TRAIN_LIMS}
VAL_LIMS = {
    'old_male_smile': 500,
    'old_male_no_smile': 500,
    'old_female_smile': 500,
    'old_female_no_smile': 500,
    'young_male_smile': 500,
    'young_male_no_smile': 500,
    'young_female_smile': 500,
    'young_female_no_smile': 500
}
val_counts = {subgroup: 0 for subgroup in VAL_LIMS}

# Read in attributes from csv
celeba_df = pd.read_csv('list_attr_celeba.csv')
count = 0 # pylint: disable=invalid-name

for f_path in celeb_paths:

    if count%10000 == 0:
        print('Now checking file number ', count)
    count += 1

    f_name = f_path.split('/')[-1]
    view = celeba_df[celeba_df['filename'] == f_name]
    age = "young" if view['Young'].item() == 1 else "old" # pylint: disable=invalid-name
    sex = "male" if view['Male'].item() == 1 else "female" # pylint: disable=invalid-name
    smile = "smile" if view['Smiling'].item() == 1 else "no_smile" # pylint: disable=invalid-name
    full_key = "_".join((age, sex, smile)) # pylint: disable=invalid-name

    if train_counts[full_key] < TRAIN_LIMS[full_key]:
        # add file to training data
        destination_path = os.path.join(TRAIN_DIR, age, sex, smile)
        shutil.move(f_path, destination_path)
        train_counts[full_key] = train_counts[full_key] + 1

    elif val_counts[full_key] < VAL_LIMS[full_key]:
        # add file to validation data
        destination_path = os.path.join(VAL_DIR, age, sex, smile)
        shutil.move(f_path, destination_path)
        val_counts[full_key] = val_counts[full_key] + 1

print("Finished creating training and validation sets.")
print("TRAIN COUNTS: ", train_counts)
print("VAL COUNTS: ", val_counts)

# Go through the eval partitions
# file and move the file into the
# test folder if possible
with open(PARTITION_FILE, encoding='utf-8') as f:
    for line in f:
        f_name, partition = line.split()
        f_path = os.path.join(CELEB_DIR, f_name)
        if partition == "2":
            if f_path in celeb_paths:
                view = celeba_df[celeba_df['filename'] == f_name]
                age = "young" if view['Young'].item() == 1 else "old" #pylint: disable=invalid-name
                sex = "male" if view['Male'].item() == 1 else "female" #pylint: disable=invalid-name
                smile = "smile" if view['Smiling'].item() == 1 else "no_smile" #pylint: disable=invalid-name
                destination = os.path.join(TEST_DIR, age, sex, smile)
                shutil.move(f_path, destination)

print("Finished making test set.")
