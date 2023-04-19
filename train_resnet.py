import os
import sys
import numpy as np
from dotenv import load_dotenv
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from mean_train import MEANS, STDEVS

load_dotenv()

assert torch.cuda.is_available(), "GPU is not available!"
print(torch.cuda.device_count())
DEVICE = f'cuda:{torch.cuda.device_count()-1}'

# From paper
BATCH_SIZE = 512
EPOCHS = 30
PEAK_LR = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PEAK_EPOCH = 2

# Other vars
DESTINATION_PATH = 'resnet.pth'
LR_INIT= 0.2 # This is just a guess based on how initial LR for CIFAR was 0.5
OUT_FEATS = 1
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
    """
    return DataLoader(datasets.ImageFolder(dirn, transform=data_transforms), batch_size=BATCH_SIZE)

TRAIN_DIR, VAL_DIR, TEST_DIR = os.getenv("TRAIN_DIR"), os.getenv("VAL_DIR"), os.getenv("TEST_DIR")
train_loader, val_loader, test_loader = loader(TRAIN_DIR), loader(VAL_DIR), loader(TEST_DIR)

model = torchvision.models.resnet18()
# overwrite the last layer of resnet to use
# one output class (later, use bce)
model.fc = nn.Linear(in_features=512, out_features=OUT_FEATS, bias=True)
model.to(DEVICE)
model.train()

optimizer = SGD(model.parameters(),
                lr=LR_INIT,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY)

# Implement a cyclic lr schedule
# credit: https://github.com/MadryLab/failure-directions/blob/d484125c5f5d0d7ec8666f5bfce9d496b2af83b9/failure_directions/src/optimizers.py#L1
iters_per_epoch = len(train_loader)
lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                [0, PEAK_EPOCH * iters_per_epoch, EPOCHS * iters_per_epoch],
                [0, 1, 0])
def get_lr(epo):
    global lr_schedule
    return lr_schedule[epo]
scheduler = lr_scheduler.LambdaLR(optimizer, get_lr)
scaler = GradScaler()
bce_loss_unreduced = nn.BCEWithLogitsLoss(reduction='none')

for epoch in range(EPOCHS):
    epoch_loss = 0
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(DEVICE)
        with autocast():
            logits = model(images).squeeze()
            print('logits: ', logits)
            temp_loss = bce_loss_unreduced(logits, labels.float().to(DEVICE))
            loss = temp_loss.mean()
            epoch_loss += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    print(f'Finished epoch {epoch + 1} with loss {epoch_loss}. Saving model.')
    torch.save(model.state_dict(), DESTINATION_PATH)
