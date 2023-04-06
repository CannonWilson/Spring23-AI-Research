from age_model import CustomAgeNetwork
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

# Custom model
MODEL_PATH = './my_age_model.pth'
model = CustomAgeNetwork()
model.load_state_dict(torch.load('./my_age_model.pth'))

data_transforms = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor()
])

val_data = datasets.ImageFolder("./data/validation", transform=data_transforms) # !!Right now, using validation data instead of test data 
val_loader = DataLoader(val_data, batch_size=1)

results = { # stores number of correct classifications for each category
    'old_female': {
        'correct': 0,
        'total': 0
    },
    'old_male': {
        'correct': 0,
        'total': 0
    },
    'young_female': {
        'correct': 0,
        'total': 0
    },
    'young_male': {
        'correct': 0,
        'total': 0
    }
}

with torch.no_grad():
    model.eval()
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader, 0):

            # Custom model 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = predicted.item()
            actual = labels.item()
            f_path = val_loader.dataset.samples[i][0]
            category = f_path.split('/')[-2] # category like 'young_male'
            if pred == actual:
                total_correct += 1
                results[category]['correct'] = results[category]['correct'] + 1
            results[category]['total'] = results[category]['total'] + 1
            total_num += 1

print('TOTAL ACCURACY: ', round(100 * total_correct/ total_num) / 100)
for key in results.keys():
    print(key.upper(), " ACCURACY: ", \
        round(100 * results[key]['correct'] / results[key]['total']) / 100)