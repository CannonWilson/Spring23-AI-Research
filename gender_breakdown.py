"""
Before proceeding, follow the instructions in
split_data.py.

This script loops through the downloaded images
and will output a description of the number of
male/female images in each subdirectory.
"""

import os

TRAIN_YOUNG_DIR = os.path.join('train', 'young')
TRAIN_OLD_DIR = os.path.join('train', 'old')
TEST_YOUNG_DIR = os.path.join('test', 'young')
TEST_OLD_DIR = os.path.join('test', 'old')

results = {
    TRAIN_YOUNG_DIR: {
        'male': 0,
        'male_%': 0,
        'female': 0,
        'female_%': 0
    },
    TRAIN_OLD_DIR: {
        'male': 0,
        'male_%': 0,
        'female': 0,
        'female_%': 0
    },
    TEST_YOUNG_DIR: {
        'male': 0,
        'male_%': 0,
        'female': 0,
        'female_%': 0
    },
    TEST_OLD_DIR: {
        'male': 0,
        'male_%': 0,
        'female': 0,
        'female_%': 0
    }
}

# Create lists of filenames for 
# images of males and of females
male_filenames = []
female_filenames = []
assert os.path.exists('./list_attr_celeba.txt'), \
    "list_attr_celeba.txt file not downloaded correctly"
with open("list_attr_celeba.txt") as f:
    file_lines = f.read().split("\n")[2:]# first two lines are not data
    for line in file_lines:
        # Male score is the 21st score on each line
        line_text = line.split(" ")
        filename = line_text[0]
        male_score = line_text[21] 
        if male_score == "1":
            male_filenames.append(filename)
        elif male_score == "-1":
            female_filenames.append(filename)

# Fill the results dict
for dir_tup in results.items():
    cur_dict = dir_tup[0]
    cur_results_dict = dir_tup[1]
    cur_dir_imgs = [filename for filename in \
        os.listdir(cur_dict) if filename[0] != "."]
    # Loop over the images in the current dir
    for img_name in cur_dir_imgs:
        if img_name in male_filenames:
            cur_results_dict['male'] = cur_results_dict['male'] + 1
        elif img_name in female_filenames:
            cur_results_dict['female'] = cur_results_dict['female'] + 1
    # Calculate percentages
    total_imgs = cur_results_dict['male'] + cur_results_dict['female']
    cur_results_dict['male_%'] = cur_results_dict['male'] / total_imgs
    cur_results_dict['female_%'] = cur_results_dict['female'] / total_imgs

print("#### RESULTS ####")
print(results)