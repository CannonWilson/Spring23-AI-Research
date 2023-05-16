"""
Edit this file to change 
the data/models being
used across the project
"""
CELEBA_DIR = "img_align_celeba"
CELEBA_HEADER = ['filename','5_o_Clock_Shadow','Arched_Eyebrows',\
                'Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips',\
                'Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair',\
                'Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses',\
                'Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male',\
                'Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard',\
                'Oval_Face','Pale_Skin','Pointy_Nose','Receding_Hairline',\
                'Rosy_Cheeks','Sideburns','Smiling','Straight_Hair','Wavy_Hair',\
                'Wearing_Earrings','Wearing_Hat','Wearing_Lipstick',\
                'Wearing_Necklace','Wearing_Necktie','Young']
CELEBA_ATTRS_CSV = 'celeba_info/list_attr_celeba.csv'
CELEBA_ATTRS_TXT = 'celeba_info/list_attr_celeba.txt'
CELEBA_PART_TXT = 'celeba_info/list_eval_partition.txt'
TRAIN_DIR = "data/train_2_corr"
VAL_DIR = "data/val_2_corr"
TEST_DIR = "data/test_2_corr"
MODEL_PATH = "resnet_models/new_resnet_2_corr.pth" # destination path if training new resnet model
IMG_WIDTH = 75
IMG_HEIGHT = 75
NUM_CORRS = 2 # Should be 1 or 2
CLIP_VIS = "ViT-B/32"
TRAIN_MEANS_1_CORR = {
    "red":  0.5016617507657585,
    "green": 0.4204056117111792,
    "blue": 0.37952040804628584,
}
TRAIN_STDEVS_1_CORR = {
    "red": 0.30475696241055944,
    "green": 0.2826617539495315,
    "blue": 0.2818736028057502,
}
TRAIN_MEANS_2_CORR = {
    "red":  0.5001126708333333,
    "green":  0.4186306559722222,
    "blue":  0.37775329148148146,
}
TRAIN_STDEVS_2_CORR = {
    "red":  0.30447378268133,
    "green":  0.2820215661490237,
    "blue":  0.2812688950031558,
}
SUBDIRS_1_CORR = ["old/male", "old/female",
                  "young/male","young/female"]
SUBDIRS_2_CORR = ["old/male/smile", "old/male/no_smile",
                "old/female/smile", "old/female/no_smile",
                "young/male/smile", "young/male/no_smile",
                "young/female/smile", "young/female/no_smile"]
TRAIN_LIMS_1_CORR = {
    'old_male': 24812,
    'old_female': 6203,
    'young_male': 6203,
    'young_female': 24812,
}
VAL_LIMS_1_CORR = {
    'old_male': 1000,
    'old_female': 1000,
    'young_male': 1000,
    'young_female': 1000,
}
TRAIN_LIMS_2_CORR = {
    'old_male_smile': 8000,
    'old_male_no_smile': 16000,
    'old_female_smile': 2000,
    'old_female_no_smile': 4000,
    'young_male_smile': 4000,
    'young_male_no_smile': 2000,
    'young_female_smile': 16000,
    'young_female_no_smile': 8000
}
VAL_LIMS_2_CORR = {
    'old_male_smile': 500,
    'old_male_no_smile': 500,
    'old_female_smile': 500,
    'old_female_no_smile': 500,
    'young_male_smile': 500,
    'young_male_no_smile': 500,
    'young_female_smile': 500,
    'young_female_no_smile': 500
}
