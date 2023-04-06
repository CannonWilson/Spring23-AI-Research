import csv
import os
import shutil
import pandas as pd
from itertools import islice
from pathlib import Path

CELEB_DIR = './img_align_celeba_3'
TRAIN_DIR = './train'
VAL_DIR = './val'

celeb_dir_path = Path(CELEB_DIR)
paths = [i.path for i in islice(os.scandir(celeb_dir_path), None)]

""" (OLD) variables for dataset with 2 correlations - male, hat """
# train_lims = {
#     'old_male_hat': 2100,
#     'old_male_no_hat': 20000-2100,
#     'old_female_hat': 300,
#     'old_female_no_hat': 5000-300,
#     'young_male_hat': 300,
#     'young_male_no_hat': 5000-300,
#     'young_female_hat': 2100,
#     'young_female_no_hat': 20000-2100
# }
# train_counts = {
#     'old_male_hat': 0,
#     'old_male_no_hat': 0,
#     'old_female_hat': 0,
#     'old_female_no_hat': 0,
#     'young_male_hat': 0,
#     'young_male_no_hat': 0,
#     'young_female_hat': 0,
#     'young_female_no_hat': 0   
# }
# val_lims = {
#     'old_male_hat': 15,
#     'old_male_no_hat': 1000-15,
#     'old_female_hat': 15,
#     'old_female_no_hat': 1000-15,
#     'young_male_hat': 15,
#     'young_male_no_hat': 1000-15,
#     'young_female_hat': 15,
#     'young_female_no_hat': 1000-15
# }
# val_counts = {
#     'old_male_hat': 0,
#     'old_male_no_hat': 0,
#     'old_female_hat': 0,
#     'old_female_no_hat': 0,
#     'young_male_hat': 0,
#     'young_male_no_hat': 0,
#     'young_female_hat': 0,
#     'young_female_no_hat': 0   
# }

""" variables for dataset with 2 correlations - male, glasses """
# c1_pri = 24000 # male/female
# c1_sec = 6000
# c2_pri = 1600 # glasses/no
# c2_sec = 400

# train_lims = {
#     'old_male_glasses': c2_pri,
#     'old_male_no_glasses': c1_pri-c2_pri,
#     'old_female_glasses': c2_sec,
#     'old_female_no_glasses': c1_sec-c2_sec,
#     'young_male_glasses': c2_sec,
#     'young_male_no_glasses': c1_sec-c2_sec,
#     'young_female_glasses': c2_pri,
#     'young_female_no_glasses': c1_pri-c2_pri
# }
# train_counts = {
#     'old_male_glasses': 0,
#     'old_male_no_glasses': 0,
#     'old_female_glasses': 0,
#     'old_female_no_glasses': 0,
#     'young_male_glasses': 0,
#     'young_male_no_glasses': 0,
#     'young_female_glasses': 0,
#     'young_female_no_glasses': 0   
# }
# val_lims = {
#     'old_male_glasses': 15,
#     'old_male_no_glasses': 1000-15,
#     'old_female_glasses': 15,
#     'old_female_no_glasses': 1000-15,
#     'young_male_glasses': 15,
#     'young_male_no_glasses': 1000-15,
#     'young_female_glasses': 15,
#     'young_female_no_glasses': 1000-15
# }
# val_counts = {
#     'old_male_glasses': 0,
#     'old_male_no_glasses': 0,
#     'old_female_glasses': 0,
#     'old_female_no_glasses': 0,
#     'young_male_glasses': 0,
#     'young_male_no_glasses': 0,
#     'young_female_glasses': 0,
#     'young_female_no_glasses': 0   
# }

""" (NEWEST) variables for dataset with 2 correlations - male, smile """
train_lims = {
    'old_male_smile': 8000,
    'old_male_no_smile': 16000,
    'old_female_smile': 2000,
    'old_female_no_smile': 4000,
    'young_male_smile': 4000,
    'young_male_no_smile': 2000,
    'young_female_smile': 16000,
    'young_female_no_smile': 8000
}
train_counts = {
    'old_male_smile': 0,
    'old_male_no_smile': 0,
    'old_female_smile': 0,
    'old_female_no_smile': 0,
    'young_male_smile': 0,
    'young_male_no_smile': 0,
    'young_female_smile': 0,
    'young_female_no_smile': 0
}
val_lims = {
    'old_male_smile': 500,
    'old_male_no_smile': 500,
    'old_female_smile': 500,
    'old_female_no_smile': 500,
    'young_male_smile': 500,
    'young_male_no_smile': 500,
    'young_female_smile': 500,
    'young_female_no_smile': 500
}
val_counts = {
    'old_male_smile': 0,
    'old_male_no_smile': 0,
    'old_female_smile': 0,
    'old_female_no_smile': 0,
    'young_male_smile': 0,
    'young_male_no_smile': 0,
    'young_female_smile': 0,
    'young_female_no_smile': 0
}


# Read in attributes from csv
celeba_df = pd.read_csv('list_attr_celeba.csv')
count = 0

for f_path in paths:

    if count%10000 == 0:
        print('Now checking file number ', count)
    count += 1

    f_name = f_path.split('/')[-1]
    view = celeba_df[celeba_df['filename'] == f_name]
    smile = view['Smiling'].item()
    young = view['Young'].item()
    male = view['Male'].item()
    age = "young" if young == 1 else "old"
    sex = "male" if male == 1 else "female"
    smile = "smile" if smile == 1 else "no_smile"
    full_key = "_".join((age, sex, smile))

    if train_counts[full_key] < train_lims[full_key]:
        # add file to training data
        destination_path = os.path.join(TRAIN_DIR, age, sex, smile)
        shutil.move(f_path, destination_path)
        train_counts[full_key] = train_counts[full_key] + 1

    elif val_counts[full_key] < val_lims[full_key]:
        # add file to validation data
        destination_path = os.path.join(VAL_DIR, age, sex, smile)
        shutil.move(f_path, destination_path)
        val_counts[full_key] = val_counts[full_key] + 1

print("TRAIN COUNTS: ", train_counts)
print("VAL COUNTS: ", val_counts)