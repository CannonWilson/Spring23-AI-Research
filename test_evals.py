import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from age_model import CustomAgeNetwork

load_dotenv()

USE_RESNET = True

# Load model
MODEL_PATH = os.getenv("MODEL_PATH")
if USE_RESNET:
    model = torchvision.models.resnet18()
else: 
    model = CustomAgeNetwork()
model.load_state_dict(torch.load(MODEL_PATH))

img_size = (os.getenv("IMG_WIDTH"), os.getenv("IMG_HEIGHT"))
assert img_size == (75, 75), "Images must be 75x75"
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

def loader(dirn):
    """ 
    Create data using torchvision 
    ImageFolder and torch DataLoader.
    Assumes batch_size=1.
    """
    return DataLoader(datasets.ImageFolder(dirn, transform=data_transforms), batch_size = 1)

TRAIN_DIR, VAL_DIR, TEST_DIR = os.getenv("TRAIN_DIR"), os.getenv("VAL_DIR"), os.getenv("TEST_DIR")
train_loader, val_loader, test_loader = loader(TRAIN_DIR), loader(VAL_DIR), loader(TEST_DIR)

def test_acc(data_loader, mode):
    """
    Loop through the data loader
    to calculate the accuracy 
    over the entire dataset.
    mode is a string in 
    {"train", "val", "test"}
    """

    # stores number of correct classifications for each subgroup
    results = {
        'old_female_no_smile': {
            'correct': 0,
            'total': 0
        },
        'old_female_smile': {
            'correct': 0,
            'total': 0
        },
        'old_male_no_smile': {
            'correct': 0,
            'total': 0
        },
        'old_male_smile': {
            'correct': 0,
            'total': 0
        },
        'young_female_no_smile': {
            'correct': 0,
            'total': 0
        },
        'young_female_smile': {
            'correct': 0,
            'total': 0
        },
        'young_male_no_smile': {
            'correct': 0,
            'total': 0
        },
        'young_male_smile': {
            'correct': 0,
            'total': 0
        }
    }

    total_correct = 0
    total_num = 0

    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(data_loader):

            # Custom model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = predicted.item()
            actual = labels.item()
            f_path = data_loader.dataset.samples[i][0]

            # Use file path to get attributes
            full_key = "_".join(f_path.split("/")[-4:-1]) # ex: old_female_no_smile
            if pred == actual:
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
