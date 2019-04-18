## learning website : http://blog.itpub.net/31562039/viewspace-2565264/

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


## prep for dataset
_task = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]
)

mnist = MNIST("data", download=True, train=True, transforms=_task)

# create data split: train, validation, test
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

# create sampler using SubsetRamdomSampler
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# create iterator for train and validation dataset
train_loader = DataLoader(mnist, batch_size=256, sampler=train_sampler)
valid_loader = DataLoader(mnist, batch_size=256, sampler=valid_sampler)



## model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.hidden
        x = F.sigmoid(x)
        x = self.output(x)
        return x

model  = Model


if __name__ == "__main__":
    pass