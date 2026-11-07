from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

model_args = {}
# random seed
model_args['seed'] = 123
# we will use batch size of 128 in Stochastic Gradient Descent (SGD) optimization of the network
model_args['batch_size'] = 128
# learning rate is how fast it will descend
model_args['lr'] = .05
# SGD momentum (default: .5) momentum is a moving average of gradients (it helps to keep direction)
model_args['momentum'] = .5
# the number of epochs is the number of times you go through the full dataset
model_args['epochs'] = 50
# logging frequency
model_args['log_interval'] = 100


class CNN(nn.Module):
    # this defines the structure of the CNN model
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layer with 5 kernels of size 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3,stride=2)
        # 5 kernels of size 3x3
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)


        # 2D dropout
        self.conv2_drop = nn.Dropout2d(p=0.25)

        # fully connected layers
        self.fc1 = nn.Linear(16 * 22 * 22 , 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        # 1st layers group
        x = self.conv1(x) # resulting in 5 feature maps each of size 13x13
        x = F.relu(x) # standard (in CNNs) ReLU activation
        # 2nd group
        x = self.conv2(x) # resulting in 10 feature maps each of size 11x11
        x = F.relu(x)
        x = self.conv2_drop(x) # dropout regularization
        # fully connected layers
        x = x.view(-1, 256) # 4 maps of 4x4 size gives 256 numbers
        x = self.fc1(x) # 256 -> 20
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # dropout is a type of regularization
        x = self.fc2(x) # 20 -> 10
        # softmax (multinomial classification) gives probabilities of each class
        return F.log_softmax(x, dim=1) # note that dim=0 is the number of samples in batch


def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    # get subsequent batches over the data in a given epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # send data tensors to GPU (or CPU)
        data, target = data.to(device), target.to(device)
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
        # Uwzględniamy rzeczywistą liczbę obrazów w batchu
        train_loss += loss.item() * len(data)
    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def test(model, device, test_loader, message, collect_predictions=False):
    model.eval()
    test_loss = 0.
    correct = 0
    # data to confusion matrix
    true_labels = []
    predicted_labels = []
    # this is just inference, we don't need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # calculate and sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of class with the max log-probability
            prediction = output.argmax(dim=1)
            # item() returns value of the given tensor
            correct += prediction.eq(target).sum().item()
            # Zbieramy etykiety tylko przy końcowym teście, nie w każdej walidacji
            if collect_predictions:
                true_labels.extend(target.cpu().tolist())
                predicted_labels.extend(prediction.cpu().tolist())
    test_loss /= len(test_loader.dataset)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if collect_predictions:
        return test_loss, true_labels, predicted_labels
    return test_loss


def plot_loss(train_loss, validation_loss, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("subsequent epochs")
    ax.set_ylabel("average loss")
    ax.plot(range(1, len(train_loss)+1), train_loss, 'o-', markersize=4, label='training')
    ax.plot(range(1, len(validation_loss)+1), validation_loss, 'o-', markersize=4, label='validation')
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)



def main():
    # Skrypt działa też uruchomiony z innego katalogu w terminalu
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Seed przed inicjalizacją modelu i tworzeniem podziału danych
    torch.manual_seed(model_args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the MINST dataset via torchvision
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(str(data_dir), train=True, download=True,
                                transform=transform)
    # we divide this data into training and validation subsets
    split_generator = torch.Generator().manual_seed(model_args['seed'])
    train_subset, validation_subset = torch.utils.data.random_split(
        mnist_train, [50000, 10000], generator=split_generator
    )
    test_subset = datasets.MNIST(str(data_dir), train=False, download=True,
                                 transform=transform)

    # define dataloaders
    loader_kwargs = {'batch_size': model_args['batch_size'],
                     'num_workers': 0,  # Stabilnie na Windows; można zmienić na 2
                     'pin_memory': device.type == 'cuda',
                     'shuffle': True}
    train_loader = torch.utils.data.DataLoader(train_subset, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(
        validation_subset, **{**loader_kwargs, 'shuffle': False}
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, **{**loader_kwargs, 'shuffle': False}
    )

    model = CNN().to(device)

    optimizer = optim.SGD(model.parameters(), lr=model_args['lr'], momentum=model_args['momentum'])
    #optimizer = optim.Adadelta(model.parameters(), lr=model_args['lr'])

    torch.manual_seed(model_args['seed'])
    train_loss = []
    validation_loss = []
    for epoch_number in range(1, model_args['epochs'] + 1):
        train_loss.append(train(model, device, train_loader, optimizer, epoch_number))
        validation_loss.append(test(model, device, validation_loader, 'Validation set'))

    _, true_labels, predicted_labels = test(
        model, device, test_loader, 'Test set', collect_predictions=True
    )
    plot_loss(train_loss, validation_loss, 'CNN model',
              plots_dir / 'CNN_loss.png')

    # =============================================================================
    # CONFUSION MATRIX
    # =============================================================================

    M = np.zeros((10, 10), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        M[true, pred] += 1

    # =============================================================================
    # NORMALIZACJA
    # =============================================================================

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
    assert M.sum() == len(test_subset), "Niepoprawna liczba próbek w macierzy!"

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
    diagonal_mask = np.eye(10, dtype=bool)

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
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")

        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))

        # Delikatna siatka pomiędzy komórkami
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Wpisanie wartości do komórek
        for i in range(10):
            for j in range(10):

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
        f"CNN — MNIST test set | Accuracy = {accuracy * 100:.2f}%",
        fontsize=17
    )

    fig.savefig(plots_dir / "CNN_confusion_matrix.png", dpi=200)
    plt.close(fig)

    # =============================================================================
    # NAJCZĘSTSZE POMYŁKI
    # =============================================================================

    M_errors = M.copy()
    np.fill_diagonal(M_errors, 0)

    errors = []
    for i in range(10):
        for j in range(10):
            if M_errors[i, j] > 0:
                errors.append((M_errors[i, j], i, j))

    errors.sort(reverse=True)
    print("\nNajczęstsze pomyłki modelu:")
    for count, true, pred in errors[:10]:
        print(f"Cyfra {true} rozpoznana jako {pred}: {count} razy")

    print(f"\nWykresy zapisano w: {plots_dir}")
    plt.show()


# Ochrona przed ponownym wykonaniem kodu przy multiprocessing na Windows
if __name__ == "__main__":
    main()
