from itertools import islice
from pathlib import Path
import os
import shutil
import pandas as pd

TEST_DIR = './test'

celeba_df = pd.read_csv('list_attr_celeba.csv')
test_paths = [i.path for i in islice(os.scandir(TEST_DIR), None)]

for f_path in test_paths:
    f_name = os.path.split(f_path)[-1]
    if not f_name.endswith(".jpg"): continue
    view = celeba_df[celeba_df['filename'] == f_name]
    smile = view['Smiling'].item()
    young = view['Young'].item()
    male = view['Male'].item()
    age = "young" if young == 1 else "old"
    sex = "male" if male == 1 else "female"
    smile = "smile" if smile == 1 else "no_smile"
    destination = os.path.join(TEST_DIR, age, sex, smile)
    shutil.move(f_path, destination)