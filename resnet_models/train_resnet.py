"""
Train a new resnet model! Make 
sure to check all of the info 
in settings.py to ensure that
the model will be trained
and saved with the desired
settings.
"""

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, lr_scheduler
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from settings import NUM_CORRS, TRAIN_MEANS_1_CORR, TRAIN_MEANS_2_CORR, \
    TRAIN_STDEVS_1_CORR, TRAIN_STDEVS_2_CORR, MODEL_PATH, \
    IMG_WIDTH, IMG_HEIGHT, TRAIN_DIR

MEANS = TRAIN_MEANS_1_CORR if NUM_CORRS == 1 else TRAIN_MEANS_2_CORR
STDEVS = TRAIN_STDEVS_1_CORR if NUM_CORRS == 1 else TRAIN_STDEVS_2_CORR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# From paper
BATCH_SIZE = 512
EPOCHS = 30
PEAK_LR = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PEAK_EPOCH = 2
OUT_FEATS = 2 # Using cross entropy loss with 2 output feats

# Other vars
print(f'Beginning training. Saving model to {MODEL_PATH}')
LR_INIT= 0.5 # This is just a guess based on how initial LR for CIFAR was 0.5 in Example notebook
img_size = (IMG_WIDTH, IMG_HEIGHT)
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(MEANS.values()), std=list(STDEVS.values()))
])
train_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR, transform=data_transforms), \
                          batch_size=BATCH_SIZE, shuffle=True)

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
# credit: https://github.com/MadryLab/failure-directions/blob/d484125c5f5d0d7ec8666f5bfce9d496b2af83b9/failure_directions/src/optimizers.py#L1 pylint:disable=line-too-long
iters_per_epoch = len(train_loader)
lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                [0, PEAK_EPOCH * iters_per_epoch, EPOCHS * iters_per_epoch],
                [0, 1, 0])
def get_lr(epo):
    """
    Simple learning rate indexer function
    because torch optim's lr_scheduler
    requires such a function as input
    """
    return lr_schedule[epo]
scheduler = lr_scheduler.LambdaLR(optimizer, get_lr)
scaler = GradScaler()
ce_loss = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with autocast():
            logits = model(images)
            loss = ce_loss(logits, labels.long())
            epoch_loss += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pred = torch.argmax(logits, dim=1)
        correct = pred == labels
        epoch_correct += correct.sum()
        epoch_total += labels.size()[0]
    acc = epoch_correct / epoch_total
    print('#### epoch: ', epoch+1,' #### ')
    print('loss: ', loss)
    print('acc: ', acc)
    torch.save(model.state_dict(), MODEL_PATH)
