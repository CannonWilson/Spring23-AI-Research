import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from age_model import CustomAgeNetwork
from mean_train import MEANS, STDEVS

load_dotenv()

NUM_CORRS = int(os.getenv("NUM_CORRS"))
assert NUM_CORRS in [1,2], "Can only handle 1-2 correlations in data"
USE_RESNET = True
BATCH_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load model
MODEL_PATH = os.getenv("MODEL_PATH")
if USE_RESNET:
    OUT_FEATS = 2
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=OUT_FEATS, bias=True)
else: 
    model = CustomAgeNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

img_size = (int(os.getenv("IMG_WIDTH")), int(os.getenv("IMG_HEIGHT")))
assert img_size == (75, 75), "Images must be 75x75"
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(MEANS.values()), std=list(STDEVS.values()))
])

def loader(dirn):
    """ 
    Create data using torchvision 
    ImageFolder and torch DataLoader.
    Assumes batch_size=1.
    """
    return DataLoader(datasets.ImageFolder(dirn, transform=data_transforms), batch_size = BATCH_SIZE)

TRAIN_DIR, VAL_DIR, TEST_DIR = os.getenv("TRAIN_DIR"), os.getenv("VAL_DIR"), os.getenv("TEST_DIR")
train_loader, val_loader, test_loader = loader(TRAIN_DIR), loader(VAL_DIR), loader(TEST_DIR)

list_1_corr = ['old_female', 'old_male', 'young_female', 'young_male']
list_2_corr = ['old_female_no_smile', 'old_female_smile', 'old_male_no_smile', \
               'old_male_smile', 'young_female_no_smile', 'young_female_smile', \
                'young_male_no_smile', 'young_male_smile']

def test_acc(data_loader, mode):
    """
    Loop through the data loader
    to calculate the accuracy 
    over the entire dataset.
    mode is a string in 
    {"train", "val", "test"}
    """

    # stores number of correct classifications for each subgroup
    corr_list = list_1_corr if NUM_CORRS == 1 else list_2_corr
    results = {subgroup: {'correct': 0, 'total': 0} for subgroup in corr_list}

    total_correct = 0
    total_num = 0

    with torch.no_grad():
        model.eval()
        epoch_correct = 0
        epoch_total = 0
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Custom model
            logits = model(images) # model(images).squeeze()
            # pred = sigmoid(logits) > 0.5
            pred = torch.argmax(logits, dim=1)
            correct = pred == labels
            epoch_correct += correct.sum()
            epoch_total += labels.size()[0]
            
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + labels.size()[0]
            for f_idx, (f_path, class_num) in enumerate(data_loader.dataset.samples[start_idx:end_idx]):
                # Use file path to get attributes
                age = "young" if "young" in f_path else "old"
                sex = "female" if "female" in f_path else "male"
                if NUM_CORRS == 1:
                    full_key = "_".join([age, sex]) # ex: old_female
                else:
                    smile = "no_smile" if "no_smile" in f_path else "smile"
                    full_key = "_".join([age,sex,smile])
                if correct[f_idx] == True:
                    total_correct += 1
                    results[full_key]['correct'] = results[full_key]['correct'] + 1
                results[full_key]['total'] = results[full_key]['total'] + 1
                total_num += 1

    print(f'TOTAL {mode.upper()} ACCURACY: ', round(100 * total_correct/ total_num) / 100)
    for key, val in results.items():
        print(key.upper(), " ACCURACY: ", \
            round(100 * val['correct'] / val['total']) / 100)

if __name__ == "__main__":
    # test_acc(train_loader, "train")
    # test_acc(val_loader, "val")
    test_acc(test_loader, "test")
