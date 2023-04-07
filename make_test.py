"""
This file will fill the test
folder using the original 
CelebA test files (if they
haven't already been put 
into the train/val folders)
"""
from itertools import islice
from pathlib import Path
import os
import shutil

TEST_DIR = './test'
CELEBA_DIR = 'img_align_celeba_3'
PARTITION_FILE = './list_eval_partition.txt'
cel_dir_path = Path(CELEBA_DIR)
celeb_paths = [i.path for i in islice(os.scandir(cel_dir_path), None)]

# Go through the eval partitions
# file and move the file into the
# test folder if possible
with open(PARTITION_FILE) as f:
    for line in f:
        f_name, partition = line.split()
        f_path = os.path.join(CELEBA_DIR, f_name)
        if partition == "2":
            if f_path in celeb_paths:
                shutil.move(f_path, TEST_DIR)
