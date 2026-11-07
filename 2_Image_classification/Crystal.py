#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from wallpaper_dataset import utils
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

model_args = {}
# random seed
model_args['seed'] = 123
# we will use batch size of 128 in Stochastic Gradient Descent (SGD) optimization of the network
model_args['batch_size'] = 512
# learning rate is how fast it will descend
model_args['lr'] = 1e-3
# SGD momentum (default: .5) momentum is a moving average of gradients (it helps to keep direction)
model_args['momentum'] = .5
# the number of epochs is the number of times you go through the full dataset
model_args['epochs'] = 50
# logging frequency
model_args['log_interval'] = 100


train_loader, val_loader, test_loader= utils.parse_dataset_classification('./wallpaper_dataset/data',
                                                             splitratio=[.7,.2,.1],
                                                             batchsize=model_args['batch_size'],
                                                             normalize=False,
                                                             downsample=False)


print(len(train_loader)) # no of train batches
print(len(val_loader)) # no of train batches
print(len(test_loader)) # no of train batches
input_size = 0


for batch_idx, (data, target) in enumerate(train_loader):

    idx = 15
    print("Przykładowe dane:", data[idx].shape)
    input_size = data[idx].shape[1]
    print("Etykieta:", target[idx])

    image = data[idx, 0].cpu().numpy()

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {target[0].item()}")
    plt.axis('off')
    plt.show()

    break


class CNN(nn.Module):
    # this defines the structure of the CNN model
    def __init__(self, input_size, kernel_size=5, CNN_masks=[2, 4], CNN_series=[2, 2], pooling=True):
        super(CNN, self).__init__()

        map_size = input_size
        self.CNN = nn.ModuleList()
        self.MLP = nn.ModuleList()

        if len(CNN_masks) != len(CNN_series):
            raise ValueError("CNN_masks != CNN_series")

        padding = int(kernel_size / 2)

        # CNN
        self.CNN.append(nn.Conv2d(1, CNN_masks[0], kernel_size=kernel_size, padding=padding))
        self.CNN.append(nn.ReLU())

        for j in range(CNN_series[0] - 1):
            self.CNN.append(nn.Conv2d(CNN_masks[0], CNN_masks[0], kernel_size=kernel_size, padding=padding))
            self.CNN.append(nn.ReLU())

        self.CNN.append(nn.MaxPool2d(kernel_size=2))
        map_size = int(map_size / 2)

        for i in range(len(CNN_masks) - 1):
            self.CNN.append(nn.Conv2d(CNN_masks[i], CNN_masks[i + 1], kernel_size=kernel_size, padding=padding))
            self.CNN.append(nn.ReLU())

            for j in range(CNN_series[i + 1] - 1):
                self.CNN.append(nn.Conv2d(CNN_masks[i + 1], CNN_masks[i + 1], kernel_size=kernel_size, padding=padding))
                self.CNN.append(nn.ReLU())

            self.CNN.append(nn.MaxPool2d(kernel_size=2))
            map_size = int(map_size / 2)

        self.CNN.append(nn.Dropout2d())

        if map_size == 0:
            raise ValueError("Mapa jest za mała – map_size po CNN wynosi 0.")
        # MLP
        self.MLP.append(nn.Linear((map_size ** 2) * CNN_masks[-1], 40))
        self.MLP.append(nn.ReLU())
        self.MLP.append(nn.Linear(40, 17))

    def forward(self, x):

        for i in range(len(self.CNN)):
            x = self.CNN[i](x)

        x = torch.flatten(x, start_dim=1)

        for i in range(len(self.MLP)):
            x = self.MLP[i](x)

        return F.log_softmax(x, dim=1)  # note that dim=0 is the number of samples in batch



def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    # get subsequent batches over the data in a given epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # send data tensors to GPU (or CPU)
        data = data.to(device)
        target = target.to(device).long()
        # this will zero out the gradients for this batch
        optimizer.zero_grad()
        # this will execute the forward() function
        output = model(data)
        # calculate the negative-log-likelihood loss
        loss = F.nll_loss(output, target, reduction='mean')
        # backpropagate the loss
        loss.backward()
        # update the model weights (with assumed learning rate)
        optimizer.step()
        if batch_idx % model_args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(model, device, test_loader, message):
    model.eval()
    test_loss = 0.
    correct = 0
    # this is just inference, we don't need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).long()
            output = model(data)
            # calculate and sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='mean')
            # get the index of class with the max log-probability
            prediction = output.argmax(dim=1)
            # item() returns value of the given tensor
            correct += prediction.eq(target).sum().item()
    test_loss /= len(test_loader)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss.cpu()

def plot_loss(train_loss, validation_loss, title, savepath):
    plt.grid(True)
    plt.xlabel("subsequent epochs")
    plt.ylabel('average loss')
    plt.plot(range(1, len(train_loss)+1), train_loss, 'o-', label='training')
    plt.plot(range(1, len(validation_loss)+1), validation_loss, 'o-', label='validation')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(Path(savepath) / ("TruePredCorelation_" + parameter), dpi=150)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(input_size=input_size, kernel_size=3, CNN_masks=[20,10,5,5], CNN_series=[3,3,3,3], pooling=True).to(device)

print(model)

optimizer = optim.Adadelta(model.parameters(), lr=model_args['lr'])

torch.manual_seed(model_args['seed'])
train_loss = []
val_loss = []
for epoch_number in range(1, model_args['epochs'] + 1):
    train_loss.append(train(model, device, train_loader, optimizer, epoch_number))
    val_loss.append(test(model, device, val_loader, 'Validation set'))

test(model, device, test_loader, 'Test set')
plot_loss(train_loss, val_loss, 'CNN model', "ResultsCrystal")