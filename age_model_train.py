import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision import datasets, transforms
from age_model import CustomAgeNetwork

NUM_EPOCHS = 50
MODEL_PATH = './smiling_age_model.pth'
TRAIN_DIR = './train'
VAL_DIR = './val'
BEST_ACC_ALLOWANCE = 0.005 # How low below best acc to still save model
EPOCH_LOSS_ALLOWANCE = 100 # How high above previous loss to still save model
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
        if isinstance(layer, nn.modules.linear.Linear):
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
    """
    Loop through every image in the
    validation dataloader to calculate 
    the accuracy over the validation set.
    """
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_imgs, val_labels in val_loader:
            val_out = model(val_imgs)
            _, predicted = torch.max(val_out.data, 1)
            val_correct += (predicted == val_labels).sum().item()
            val_total += data.size(0)
    val_acc = val_correct / val_total
    print(f'Test accuracy: {val_acc}')
    return accuracy

def train(epochs, early_stop):
    """
    Train the model!
    """
    best_accuracy = 0.0
    epochs_without_improvement = 0
    prev_epoch_loss = np.inf
    print("Beginning training!")
    for epoch_idx in range(epochs):
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
        if accuracy > best_accuracy - BEST_ACC_ALLOWANCE and \
            epoch_loss < prev_epoch_loss+EPOCH_LOSS_ALLOWANCE:

            if accuracy > best_accuracy:
                best_accuracy = accuracy
            prev_epoch_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print('Saved model. Accuracy: ', accuracy)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop:
                print(f"Reached {early_stop} \
                    epochs without improvement in accuracy \
                    on test set. Stopping early at epoch \
                        {epoch_idx + 1}.")
                break

if __name__ == "__main__":
    train(NUM_EPOCHS, 10)
