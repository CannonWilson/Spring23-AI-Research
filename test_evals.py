"""
Loop through the test images to 
find model accuracy on each 
subgroup. This file assumes
the model being used is one 
of the ResNet models. 
"""

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision import datasets, transforms
from settings import NUM_CORRS, MODEL_PATH, IMG_WIDTH, IMG_HEIGHT, TRAIN_MEANS_1_CORR, \
    TRAIN_MEANS_2_CORR, TRAIN_STDEVS_1_CORR, TRAIN_STDEVS_2_CORR, TRAIN_DIR, \
    VAL_DIR, TEST_DIR, TRAIN_LIMS_1_CORR, TRAIN_LIMS_2_CORR

BATCH_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda() # pylint:disable=unnecessary-lambda-assignment
else:
    map_location='cpu'

# Load ResNet model
OUT_FEATS = 2
model = torchvision.models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=OUT_FEATS, bias=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
model.to(DEVICE)

MEANS = TRAIN_MEANS_1_CORR if NUM_CORRS == 1 else TRAIN_MEANS_2_CORR
STDEVS = TRAIN_STDEVS_1_CORR if NUM_CORRS == 1 else TRAIN_STDEVS_2_CORR
img_size = (IMG_WIDTH, IMG_HEIGHT)
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(MEANS.values()), std=list(STDEVS.values()))
])

def loader(dirn):
    """ 
    Create data using torchvision 
    ImageFolder and torch DataLoader.
    For evaluation only, since this 
    sets shuffle as False.
    """
    return DataLoader(datasets.ImageFolder(dirn, transform=data_transforms), \
                      batch_size = BATCH_SIZE, shuffle=False)

def test_acc(data_loader, mode):
    """
    Loop through the data loader
    to calculate the accuracy 
    over the entire set of images.
    mode is a string in 
    {"train", "val", "test"}
    """

    # results stores 'correct' and 'total' for each subgroup
    # uses the training limits dictionaries from settings to get
    # keys (i.e. 'old_female_smiling')
    results = {subgroup: {'correct': 0,'total': 0} for subgroup in TRAIN_LIMS_1_CORR} \
            if NUM_CORRS == 1 else \
            {subgroup: {'correct': 0,'total': 0} for subgroup in TRAIN_LIMS_2_CORR}

    total_correct, total_num = 0,0
    with torch.no_grad():
        model.eval()
        epoch_correct, epoch_total = 0,0
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Custom model
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            correct = pred == labels
            epoch_correct += correct.sum()
            epoch_total += labels.size()[0]
            
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            for f_idx, (f_path, class_num) in enumerate(data_loader.dataset.samples[start_idx:end_idx]):
                # Use file path to get attributes val_orig/old/female
                sex = "female" if "female" in f_path else "male"
                age = "young" if "young" in f_path else "old"
                full_key = "_".join([age, sex]) # ex: old_female
                if NUM_CORRS == 2:
                    smiling = "no_smile" if "no_smile" in f_path else "smile"
                    full_key = "_".join([age,sex,smiling])
                if correct[f_idx]:
                    total_correct += 1
                    results[full_key]['correct'] = results[full_key]['correct'] + 1
                results[full_key]['total'] = results[full_key]['total'] + 1
                total_num += 1

    print(f'TOTAL {mode.upper()} ACCURACY: ', round(100 * total_correct/ total_num) / 100)
    for key, val in results.items():
        print(key.upper(), " ACCURACY: ", \
            round(100 * val['correct'] / val['total']) / 100)

if __name__ == "__main__":
    print("NUM_CORRS: ", NUM_CORRS)
    # test_acc(loader(TRAIN_DIR), "train")
    test_acc(loader(VAL_DIR), "val")
    test_acc(loader(TEST_DIR), "test")
