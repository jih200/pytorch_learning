import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils as utils
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


## prepare the training and test datasets CIFAR 10
def dataloader(root, num_workers):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

## show some training images
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

## define a convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## define a loss function and optimizer

## train the network
def train(net, epochs, trainloader, save=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the input
            inputs, labels = data

            # zero the parameter gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if i % 2000 == 1999:
                if save:
                    name = "./checkpoint_"+str(i+1)
                    torch.save(net.state_dict(), name)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished training !')

def test(net, checkpoint, testloader, classes):
    net.state_dict(torch.load(checkpoint))
    net.eval()

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print("Ground truth: ", " ".join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print("Predicted: ", " ".join('%5s' % classes[predicted[j]] for j in range(4)))

    imshow(torchvision.utils.make_grid(images))


if __name__ == "__main__":
    net = Net()
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # imshow(torchvision.utils.make_grid(images))
    root = './data'
    trainloader, testloader, classes =  dataloader(root, 2)
    train(net, 2, trainloader, True)
    checkpoint = './checkpoint_12000'
    test(net, checkpoint, testloader, classes)