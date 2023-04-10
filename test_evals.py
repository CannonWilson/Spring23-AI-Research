from age_model import CustomAgeNetwork
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

# Custom model
MODEL_PATH = "(old)my_age_model.pth" # Not smiling: './my_age_model.pth'
model = CustomAgeNetwork()
model.load_state_dict(torch.load(MODEL_PATH))

data_transforms = transforms.Compose([
    transforms.Resize((82, 100)),
    transforms.ToTensor()
])

val_data = datasets.ImageFolder("./val", transform=data_transforms) # !!Right now, using validation data instead of test data 
val_loader = DataLoader(val_data, batch_size=1)
train_data = datasets.ImageFolder("./train", transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=1)
test_data = datasets.ImageFolder("./test", transform=data_transforms)
test_loader = DataLoader(test_data, batch_size=1)

def test_acc(data_loader, mode):

    results = { # stores number of correct classifications for each category
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
            full_key = "_".join(f_path.split("/")[2:-1]) # ex: old_female_no_smile
            if pred == actual:
                total_correct += 1
                results[full_key]['correct'] = results[full_key]['correct'] + 1
            results[full_key]['total'] = results[full_key]['total'] + 1
            total_num += 1


    print(f'TOTAL {mode.upper()} ACCURACY: ', round(100 * total_correct/ total_num) / 100)
    for key in results.keys():
        print(key.upper(), " ACCURACY: ", \
            round(100 * results[key]['correct'] / results[key]['total']) / 100)
        
if __name__ == "__main__":
    # test_acc(train_loader, "train")
    # test_acc(val_loader, "val")
    test_acc(test_loader, "test")