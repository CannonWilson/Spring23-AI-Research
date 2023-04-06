"""
Before proceeding, download the aligned
CelebA dataset available here: 
https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

and the corresponding attributes here:
https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ

This script uses the img_align_celeba
directory and the list_attr_celeba.txt
files to move the CelebA images 
into two new directories: `train`
and `test`, each with two subdirectories:
`young` and `old`
"""

import os
import random
import shutil

# Create the directories and subdirectories
# if they don't already exist
TRAIN_DIR = './train'
TEST_DIR = './test'
if not os.path.exists(TRAIN_DIR): 
    os.mkdir(TRAIN_DIR)
    os.mkdir(os.path.join(TRAIN_DIR, 'young'))
    os.mkdir(os.path.join(TRAIN_DIR, 'old'))
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)
    os.mkdir(os.path.join(TEST_DIR, 'young'))
    os.mkdir(os.path.join(TEST_DIR, 'old'))

# Raise an error if there are already 
# images in any of the folders.
num_young_train_imgs = len([file for file in \
    os.listdir(os.path.join(TRAIN_DIR, 'young')) \
        if file[0] != "."])
num_old_train_imgs = len([file for file in \
    os.listdir(os.path.join(TRAIN_DIR, 'old')) \
        if file[0] != "."])
num_young_test_imgs = len([file for file in \
    os.listdir(os.path.join(TEST_DIR, 'young')) \
        if file[0] != "."])
num_old_test_imgs = len([file for file in \
    os.listdir(os.path.join(TEST_DIR, 'old')) \
        if file[0] != "."])
if num_young_train_imgs > 0 or num_old_train_imgs > 0 \
    or num_young_test_imgs > 0 or num_old_test_imgs > 0:
    raise Exception("At least one of the folders for \
        the train/test classes is not empty. Please \
            empty all folders before running this script.")

# Loop through list_attr_celeba.txt
# file to get lists of filenames
# that are for images of 
# either young or old people
young_filenames = []
old_filenames = []
assert os.path.exists('./list_attr_celeba.txt'), \
    "list_attr_celeba.txt file not downloaded correctly"
with open("list_attr_celeba.txt") as f:
    file_lines = f.read().split("\n")[2:] # first two lines are not data
    for line in file_lines:
        # Young score is the last score on each line
        line_text = line.split(" ")
        filename = line_text[0]
        young_score = line_text[-1] 
        if young_score == "1":
            young_filenames.append(filename)
        elif young_score == "-1":
            old_filenames.append(filename)

# Move the files into the young/old sub-
# directories. Files are moved instead 
# of copied in order to save memory.
# This implementation limits the number
# of young_files to the number of old_files.
CELEBA_DIR = 'img_align_celeba'
assert os.path.exists(os.path.join(CELEBA_DIR, '000001.jpg')), \
    CELEBA_DIR + "directory not installed correctly"
PERCENT_TEST = 30
celeba_img_paths = [img_path for img_path in \
    os.listdir(CELEBA_DIR) if \
        img_path[0] != "."]
YOUNG_FILECOUNT_LIMIT = len(old_filenames)
for young_file in young_filenames[:YOUNG_FILECOUNT_LIMIT]:
    rand_num = random.uniform(0, 100)
    if os.path.exists(os.path.join(CELEBA_DIR, young_file)):
        if rand_num < PERCENT_TEST: # add to test/young
            if not os.path.exists(os.path.join(TEST_DIR, 'young', young_file)):
                shutil.move(os.path.join(CELEBA_DIR, young_file), \
                    os.path.join(TEST_DIR, 'young'))
        else: # add to train/young
            if not os.path.exists(os.path.join(TRAIN_DIR, 'young', young_file)):
                shutil.move(os.path.join(CELEBA_DIR, young_file), \
                    os.path.join(TRAIN_DIR, 'young'))
for old_file in old_filenames:
    rand_num = random.uniform(0, 100)
    if os.path.exists(os.path.join(CELEBA_DIR, old_file)):
        if rand_num < PERCENT_TEST: # add to test/old
            if not os.path.exists(os.path.join(TEST_DIR, 'old', old_file)):
                shutil.move(os.path.join(CELEBA_DIR, old_file), \
                    os.path.join(TEST_DIR, 'old'))
        else: # add to train/old
            if not os.path.exists(os.path.join(TRAIN_DIR, 'old', old_file)):
                shutil.move(os.path.join(CELEBA_DIR, old_file), \
                    os.path.join(TRAIN_DIR, 'old'))

"""
If this script runs successfully, you should 
have approximately these file counts:
./img_align_celeba: 110,869 (leftovers)
./train/young: 32,235
./train/old: 32,117
./test/young: 13,630
./test/old: 13,748
"""