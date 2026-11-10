#!/usr/bin/env python3


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from wallpaper_dataset import utils
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time

plt.rcParams.update({'font.size': 14})


print(np.__version__, flush=True)
print(cv2.__version__, flush=True)


model_args = {}
# random seed
model_args['seed'] = 123
# we will use batch size of 128 in Stochastic Gradient Descent (SGD) optimization of the network
model_args['batch_size'] = 128
# learning rate is how fast it will descend
model_args['lr'] = 1e-3
# SGD momentum (default: .5) momentum is a moving average of gradients (it helps to keep direction)
model_args['momentum'] = .5
# the number of epochs is the number of times you go through the full dataset
model_args['epochs'] = 2
# logging frequency
model_args['log_interval'] = 10
#saved checkpoint
model_args['checkpoint_path'] = "CNN_checkpoint_test"
#checkpoint loaded at the start (MUST HAVE THE SAME ARCHITECTURE)
model_args['load_checkpoint_path'] = ""


train_loader, val_loader, test_loader= utils.parse_dataset_classification('./wallpaper_dataset/data',
                                                             splitratio=[.7,.2,.1],
                                                             batchsize=model_args['batch_size'],
                                                             normalize=False,
                                                             downsample=True,
                                                             preloaddata=True)


print(len(train_loader)) # no of train batches
print(len(val_loader)) # no of train batches
print(len(test_loader), flush=True) # no of train batches
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

        self.CNN.append(nn.AdaptiveAvgPool2d((8, 8)))
        self.CNN.append(nn.Dropout2d(p=0.2))

        if map_size == 0:
            raise ValueError("Mapa jest za mała – map_size po CNN wynosi 0.")
        # MLP
        self.MLP.append(nn.Linear((8 ** 2) * CNN_masks[-1], 256))
        self.MLP.append(nn.ReLU())
        self.MLP.append(nn.Linear(256, 128))
        self.MLP.append(nn.ReLU())
        self.MLP.append(nn.Linear(128, 17))

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
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(model, device, test_loader, message):
    model.eval()
    test_loss = 0.
    correct = 0
    pred_labels = []
    true_labels = []
    # this is just inference, we don't need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).long()
            output = model(data)
            #pred = output.argmax(dim=1)

            #print("Predictions:", pred[:20])
            #print("Targets:", target[:20])
            # calculate and sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='mean')
            # get the index of class with the max log-probability
            prediction = output.argmax(dim=1)
            pred_labels.append(prediction.cpu())
            true_labels.append(target.cpu())
            # item() returns value of the given tensor
            correct += prediction.eq(target).sum().item()
    test_loss /= len(test_loader)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)

    return (
        test_loss.cpu(),
        torch.cat(pred_labels).numpy(),
        torch.cat(true_labels).numpy()
    )

def plot_loss(train_loss, validation_loss, title, savepath):
    plt.grid(True)
    plt.xlabel("subsequent epochs")
    plt.ylabel('average loss')
    plt.plot(range(1, len(train_loss)+1), train_loss, 'o-', label='training')
    plt.plot(range(1, len(validation_loss)+1), validation_loss, 'o-', label='validation')
    plt.legend()
    plt.title(title)
    plt.savefig(Path(savepath) / ("LossPlot.png"), dpi=150)
    plt.show()


def plot_conf_mat(true_labels, predicted_labels, savepath):
    M = np.zeros((17, 17), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        M[true, pred] += 1


    # Suma każdego wiersza = liczba obrazów danej cyfry
    row_sums = M.sum(axis=1, keepdims=True)

    # Normalizacja do przedziału [0, 1], a następnie do procentów
    M_normalized = np.divide(
        M, row_sums,
        out=np.zeros_like(M, dtype=float),
        where=row_sums != 0
    )
    M_percent = M_normalized * 100

    # Sprawdzenie poprawności
    accuracy = np.trace(M) / M.sum()
    assert M.sum() == len(predicted_labels), "Niepoprawna liczba próbek w macierzy!"

    print(f"Liczba obrazów testowych: {M.sum()}")
    print(f"Accuracy z confusion matrix: {accuracy * 100:.2f}%")
    print("Suma wierszy po normalizacji [%]:", M_percent.sum(axis=1))

    # =============================================================================
    # WIZUALIZACJA
    # =============================================================================

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)

    matrices = [M, M_percent]
    titles = ["Confusion matrix — counts", "Normalized confusion matrix — %"]
    colorbar_labels = ["Number of errors", "Percentage [%]"]

    # Maska przekątnej
    diagonal_mask = np.eye(17, dtype=bool)

    # Kopia mapy kolorów z białym kolorem dla zamaskowanych wartości
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")

    for ax, matrix, title, cbar_label, is_percent in zip(
            axes, matrices, titles, colorbar_labels, [False, True]):
        # Wykluczamy przekątną ze skali kolorów
        matrix_masked = np.ma.array(matrix, mask=diagonal_mask)

        # Maksymalna wartość poza przekątną
        max_error = matrix[~diagonal_mask].max()
        vmax = max(max_error, 1)

        im = ax.imshow(
            matrix_masked,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=vmax
        )

        ax.set_title(title, fontsize=15, pad=15)
        ax.set_xlabel("Predicted group")
        ax.set_ylabel("True group")

        ax.set_xticks(np.arange(17))
        ax.set_xticklabels(list(utils.sorted_groups.keys()), rotation=90, ha='right')
        ax.set_yticks(np.arange(17))
        ax.set_yticklabels(list(utils.sorted_groups.keys()), rotation=0, ha='right')

        # Delikatna siatka pomiędzy komórkami
        ax.set_xticks(np.arange(-0.5, 17, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 17, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Wpisanie wartości do komórek
        for i in range(17):
            for j in range(17):

                if M[i, j] == 0 and i != j:
                    continue

                value = matrix[i, j]

                if is_percent:
                    label = f"{value:.1f}%"
                else:
                    label = f"{int(value)}"

                # Na przekątnej zawsze czarny tekst
                if i == j:
                    text_color = "black"
                else:
                    text_color = "white" if value > 0.5 * vmax else "black"

                ax.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color
                )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label=cbar_label
        )

    fig.suptitle(
        f"CNN — Wallpaper test set | Accuracy = {accuracy * 100:.2f}%",
        fontsize=17
    )

    fig.savefig(Path(savepath) / ("CNN_confusion_matrix.png"), dpi=150)
    fig.show()
    plt.close(fig)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(input_size=input_size, kernel_size=3, CNN_masks=[16,64,128], CNN_series=[1,1,1], pooling=True).to(device)

print(model)

if model_args['load_checkpoint_path'] != "":
    model.load_state_dict(torch.load(model_args['load_checkpoint_path'], map_location=device))

#optimizer = optim.SGD(model.parameters(), lr=model_args['lr'], momentum=model_args['momentum'])
optimizer = optim.Adam(model.parameters(), lr=model_args['lr'])

torch.manual_seed(model_args['seed'])
train_loss = []
val_loss = []
best_val_loss = float("inf")
for epoch_number in range(1, model_args['epochs'] + 1):
    start = time.perf_counter()
    train_loss.append(train(model, device, train_loader, optimizer, epoch_number))
    loss, _, __ = test(model, device, val_loader, 'Validation set')
    val_loss.append(loss)
    end = time.perf_counter()

    print(f"Epoch time: {end - start:.4f} seconds")

    if val_loss[-1] < best_val_loss:
        best_val_loss = val_loss[-1]
        torch.save(model.state_dict(), model_args['checkpoint_path'])
        print(f"  -> new best val loss, checkpoint saved to {model_args['checkpoint_path']}")


model.load_state_dict(torch.load(model_args['checkpoint_path'], map_location=device))

_, predicted_labels, true_labels = test(model, device, test_loader, 'Test set')
plot_loss(train_loss, val_loss, 'CNN model', "ResultsCrystal")
plot_conf_mat(true_labels, predicted_labels, "ResultsCrystal")