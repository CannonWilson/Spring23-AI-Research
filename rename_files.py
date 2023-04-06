import os
import fileinput

BASE_DIR = './data/validation'
OLD_DIR = './data/validation/old'
YOUNG_DIR = './data/validation/young'
TEXT_DIR = './list_attr_celeba.txt'

old_files = [f for f in os.listdir(OLD_DIR) if f.endswith('.jpg')]
young_files = [f for f in os.listdir(YOUNG_DIR) if f.endswith('.jpg')]

for line in fileinput.input(TEXT_DIR):
    items = line.split()
    f_name = items[0]
    if f_name in old_files:
        new_f_name = "_".join(items[1:]) + "_" + f_name
        os.rename(OLD_DIR+"/"+f_name, \
                    OLD_DIR+"/"+new_f_name)
    elif f_name in young_files:
        new_f_name = "_".join(items[1:]) + "_" + f_name
        os.rename(YOUNG_DIR+"/"+f_name, \
                    YOUNG_DIR+"/"+new_f_name)