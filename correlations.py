# Convert list_attr_celeba.txt into .csv
"""
import csv
with open('list_attr_celeba.txt') as f_txt:
    with open('list_attr_celeba.csv', 'w') as f_csv:
        writer = csv.writer(f_csv)
        f_txt.readline() # throw away the first line, it's just a number
        while True:
            line = f_txt.readline()
            if not line:
                break
            writer.writerow(line.split())
"""

# Convert training data into .csv
import csv
import os
import pandas as pd

celeba_df = pd.read_csv('list_attr_celeba.csv')

counts = {
    ''
}

def get_attrs(files):
    total = 0
    attrs = {
        'young': 0,
        'old': 0,
        'young male': 0,
        'young female': 0,
        'old male': 0,
        'old female': 0,
        'young male w/ glasses': 0,
        'young female w/ glasses': 0,
        'old male w/ glasses': 0,
        'old female w/ glasses': 0,
        'young male w/ hat': 0,
        'young female w/ hat': 0,
        'old male w/ hat': 0,
        'old female w/ hat': 0,
        'young male w/ glasses and hat': 0,
        'young female w/ glasses and hat': 0,
        'old male w/ glasses and hat': 0,
        'old female w/ glasses and hat': 0,
    }
    for f_name in files: 
        total += 1
        view = celeba_df[celeba_df['filename'] == f_name]

        glasses = view['Eyeglasses'].item()
        hat = view['Wearing_Hat'].item()
        young = view['Young'].item()
        male = view['Male'].item()

        if young == 1:
            attrs['young'] = attrs['young'] + 1
            if male == 1:
                attrs['young male'] = attrs['young male'] + 1
                if glasses == 1:
                    attrs['young male w/ glasses'] = attrs['young male w/ glasses'] + 1
                if hat == 1:
                    attrs['young male w/ hat'] = attrs['young male w/ hat'] + 1
                if glasses == 1 and hat == 1:
                    attrs['young male w/ glasses and hat'] = attrs['young male w/ glasses and hat'] + 1
            if male == -1:
                attrs['young female'] = attrs['young female'] + 1
                if glasses == 1:
                    attrs['young female w/ glasses'] = attrs['young female w/ glasses'] + 1
                if hat == 1:
                    attrs['young female w/ hat'] = attrs['young female w/ hat'] + 1
                if glasses == 1 and hat == 1:
                    attrs['young female w/ glasses and hat'] = attrs['young female w/ glasses and hat'] + 1

        if young == -1:
            attrs['old'] = attrs['old'] + 1
            if male == 1:
                attrs['old male'] = attrs['old male'] + 1
                if glasses == 1:
                    attrs['old male w/ glasses'] = attrs['old male w/ glasses'] + 1
                if hat == 1:
                    attrs['old male w/ hat'] = attrs['old male w/ hat'] + 1
                if glasses == 1 and hat == 1:
                    attrs['old male w/ glasses and hat'] = attrs['old male w/ glasses and hat'] + 1
            if male == -1:
                attrs['old female'] = attrs['old female'] + 1
                if glasses == 1:
                    attrs['old female w/ glasses'] = attrs['old female w/ glasses'] + 1
                if hat == 1:
                    attrs['old female w/ hat'] = attrs['old female w/ hat'] + 1
                if glasses == 1 and hat == 1:
                    attrs['old female w/ glasses and hat'] = attrs['old female w/ glasses and hat'] + 1

    return attrs
    


DIRS = ['./data/train/old/old_male',
        './data/train/old/old_female',
        './data/train/young/young_male',
        './data/train/young/young_female']
for dir in DIRS:
    files = [f for f in os.listdir(dir) if f[0] != '.']
    subgroup = dir.split('/')[-1]
    print(f'For subgroup {subgroup}, attributes: {get_attrs(files)}')

# Output:
"""
For subgroup old_male, attributes: {'old': 24812, 'old male': 24812, 'old male w/ glasses': 5266, 'old male w/ hat': 2092, 'old male w/ glasses and hat': 504}
For subgroup old_female, attributes: {'old': 6203, 'old female': 6203, 'old female w/ glasses': 387, 'old female w/ hat': 141, 'old female w/ glasses and hat': 32}
For subgroup young_male, attributes: {'young': 6203, 'young male': 6203, 'young male w/ glasses': 429, 'young male w/ hat': 484, 'young male w/ glasses and hat': 55}
For subgroup young_female, attributes: {'young': 24812, 'young female': 24812, 'young female w/ glasses': 455, 'young female w/ hat': 624, 'young female w/ glasses and hat': 37}
"""