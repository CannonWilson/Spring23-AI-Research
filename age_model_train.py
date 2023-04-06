from age_model import CustomAgeNetwork
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights, shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import sys

NUM_EPOCHS = 50
MODEL_PATH = './smiling_age_model.pth'
TRAIN_DIR = './train'
VAL_DIR = './val'
USE_PRETRAINED = False
if USE_PRETRAINED:
    model = shufflenet_v2_x0_5(weights = ShuffleNet_V2_X0_5_Weights.DEFAULT)
    # Overwrite last fc layer in pretrained model for new linear layers
    model.fc = nn.Sequential(nn.Linear(1024, 256),
                            nn.ReLU(),
                            nn.Linear(256, 2),
                            nn.Softmax())
    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze my fc layers
    for layer in model.fc:
        if type(layer) == nn.modules.linear.Linear:
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
    data_transforms = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()
    # Only optimize parameters that aren't frozen
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0001)
else:
    model = CustomAgeNetwork()
    data_transforms = transforms.Compose([
        transforms.Resize((82, 100)),
        transforms.ToTensor()
    ])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

loss_fn = nn.CrossEntropyLoss()
train_data = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms)
val_data = datasets.ImageFolder(VAL_DIR, transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

def val_accuracy():
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for data, labels in val_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_num += data.size(0)
    accuracy = total_correct / total_num
    print(f'Test accuracy: {accuracy}')
    return accuracy

best_accuracy = 0.0
early_stop_threshold = 10
epochs_without_improvement = 0
prev_epoch_loss = np.inf
print("Beginning training!")
for epoch_idx in range(NUM_EPOCHS):
    print("----------")
    epoch_loss = 0.0
    for data, labels in train_loader:
        optimizer.zero_grad() # zero out gradients
        output = model(data) # get model output on data
        labels = F.one_hot(labels, num_classes = 2).float()
        loss = loss_fn(output, labels) # calculate loss
        loss.backward() # BPROP to calc gradients
        optimizer.step() # update weights
        epoch_loss += loss.item() * data.size(0) # add loss to running total
    print(f'Epoch: {epoch_idx + 1} | Loss: {epoch_loss}')
    accuracy = val_accuracy()
    if accuracy > best_accuracy - 0.005 and epoch_loss < prev_epoch_loss+100:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        prev_epoch_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print('Successfully saved model. Accuracy: ', accuracy)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stop_threshold:
            print(f"Reached {early_stop_threshold} \
                epochs without improvement in accuracy \
                on test set. Stopping early at epoch \
                    {epoch_idx + 1}.")
            sys.exit()