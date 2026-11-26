"""Wizualizacja CNN dla MNIST. Uruchom: python analiza_mnist.py --weights CNN_best.pth"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

MEAN, STD = 0.1307, 0.3081
# Ścieżkę można też zmienić tutaj przy uruchamianiu przez Run Python File.
WEIGHTS = Path(__file__).with_name("CNN_best.pth")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(10 * 11 * 11, 20)
        self.fc2 = nn.Linear(20, 10)

    def features(self, x):
        first = F.relu(self.conv1(x))
        return first, self.conv2_drop(F.relu(self.conv2(first)))

    def forward(self, x):
        _, second = self.features(x)
        x = F.relu(self.fc1(torch.flatten(second, 1)))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            state = state[key]
            break
    model = CNN()
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    return model.to(device).eval().requires_grad_(False)


def save(fig, path):
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def tile(ax, data, title, **kwargs):
    image = ax.imshow(data, interpolation="nearest", **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_filters(weights, folder):
    fig, axes = plt.subplots(3, 2, figsize=(6, 8), layout="constrained")
    limit = max(abs(weights).max(), 1e-9)
    for i, (ax, kernel) in enumerate(zip(axes.flat, weights)):
        im = tile(ax, kernel, f"Filtr {i+1}", cmap="RdBu_r", vmin=-limit, vmax=limit)
        for (r, c), val in np.ndenumerate(kernel):
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > .55*limit else "black")
    axes.flat[-1].axis("off")
    bar = axes.flat[-1].inset_axes([.16, .42, .68, .10])
    fig.colorbar(im, cax=bar, orientation="horizontal", label="Waga")
    fig.suptitle("Conv1  •  pięć masek 3 × 3")
    save(fig, folder / "maski_conv1.png")


OBJECTIVES = {
    "centrum": "lokalna odpowiedź przed ReLU",
    "srednia_relu": "średnia przestrzenna po ReLU",
}


def responses(model, pixels, layer, objective, warmup=False):
    first = model.conv1((pixels - MEAN) / STD)
    raw = first if layer == "conv1" else model.conv2(F.relu(first))
    if objective == "centrum":
        return raw[:, :, raw.shape[2] // 2, raw.shape[3] // 2]

    score = F.relu(raw).mean(dim=(2, 3))
    if warmup:
        # Przy całkowicie wygaszonym kanale podnoś najsilniejszą pozycję
        # przed ReLU, aż pojawi się dodatnia odpowiedź.
        return torch.where(score > 0, score, raw.amax(dim=(2, 3)))
    return score


def maximize(model, layer, steps, device, objective):
    channels = getattr(model, layer).out_channels
    # Wspólny punkt startowy dla wszystkich kanałów w danej warstwie.
    # Bez paddingu wycinek pola recepcyjnego daje dokładnie jeden wynik.
    # Współdzielone wagi dają taką samą odpowiedź jak w centrum pełnej mapy.
    patch_size = 3 if layer == "conv1" else 7
    size = patch_size if objective == "centrum" else 28
    noise = torch.rand(1, 1, size, size, device=device)
    pixels = noise.repeat(channels, 1, 1, 1).requires_grad_()
    start = responses(model, pixels, layer, objective).detach()
    best = pixels.detach().clone()
    best_score = start.diag().clone()
    optimizer = torch.optim.Adam([pixels], lr=.04)
    for _ in range(steps):
        optimizer.zero_grad()
        (-responses(model, pixels, layer, objective, warmup=True).diag().sum()).backward()
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(0, 1)
            score = responses(model, pixels, layer, objective).diag()
            improved = score > best_score
            best[improved] = pixels[improved]
            best_score = torch.maximum(score, best_score)
    final = responses(model, best, layer, objective)
    return best[:, 0].cpu().numpy(), start.cpu().numpy(), final.cpu().numpy()


def plot_optimization(layer, patterns, start, final, folder, objective):
    n = len(patterns)
    fig, axes = plt.subplots(n//5, 5, squeeze=False, figsize=(10, 2.3*(n//5)+.4),
                             layout="constrained")
    for i, ax in enumerate(axes.flat):
        tile(ax, patterns[i], f"Cel {i+1}", cmap="gray", vmin=0, vmax=1)
    fig.suptitle(f"{layer}  •  {OBJECTIVES[objective]}  •  {patterns.shape[-1]} × {patterns.shape[-1]}")
    save(fig, folder / f"{layer}_obrazy.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
    gain = final - start
    for ax, matrix, title in zip(axes, [final, gain], ["Po optymalizacji", "Zmiana względem startu"]):
        limit = max(abs(matrix).max(), 1e-9)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if r == c else "normal",
                    color="white" if abs(value) > .55*limit else "black")
        ax.set(xticks=range(n), yticks=range(n), xticklabels=range(1, n+1),
               yticklabels=range(1, n+1), xlabel="Mierzony kanał", ylabel="Kanał docelowy obrazu",
               title=title)
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=.75)
    fig.suptitle(f"{layer}  •  {OBJECTIVES[objective]}")
    save(fig, folder / f"{layer}_macierze.png")
    for suffix, values in [("odpowiedz", final), ("przyrost", gain)]:
        np.savetxt(folder / f"{layer}_{suffix}.csv", values, delimiter=",", fmt="%.7g",
                   header=",".join(f"kanal_{i+1}" for i in range(n)), comments="")


@torch.no_grad()
def plot_examples(model, dataset, device, folder):
    # Pierwsze dwa wystąpienia każdej etykiety w oryginalnej kolejności testowej.
    ids = torch.stack([torch.where(dataset.targets == d)[0][:2] for d in range(10)])
    pixels = dataset.data[ids.flatten()].float().unsqueeze(1).to(device) / 255
    x = (pixels - MEAN) / STD
    maps = [a.cpu().numpy() for a in model.features(x)]
    pred = model(x).argmax(1).tolist()
    limits = [max(a.max(), 1e-9) for a in maps]
    for digit in range(10):
        fig = plt.figure(figsize=(14, 7.8), layout="constrained")
        grid = fig.add_gridspec(4, 9, width_ratios=[1.3, .18, 1, 1, 1, 1, 1, .15, .14])
        bars = [None, None]
        for example in range(2):
            i = 2*digit + example
            block = 2*example
            tile(fig.add_subplot(grid[block:block+2, 0]), pixels[i, 0].cpu(),
                 f"Test #{ids[digit, example].item()}\nPredykcja {pred[i]}", cmap="gray", vmin=0, vmax=1)
            # Każda kolumna zawiera conv1 nad dwoma kanałami conv2.
            outer = grid[block:block+2, 2:7].subgridspec(3, 5)
            for l, activations in enumerate(maps):
                for c, channel in enumerate(activations[i]):
                    row = 0 if l == 0 else 1 + c//5
                    ax = fig.add_subplot(outer[row, c % 5])
                    bars[l] = tile(ax, channel, f"Conv{l+1} · {c+1}",
                                   cmap="magma", vmin=0, vmax=limits[l])
        for l, im in enumerate(bars):
            fig.colorbar(im, cax=fig.add_subplot(grid[2*l:2*l+2, 8]),
                         label=f"Conv{l+1} po ReLU")
        fig.suptitle(f"Cyfra {digit}  •  dwa przykłady z MNIST test")
        save(fig, folder / f"cyfra_{digit}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--data", type=Path, default=Path("data"), help="Katalog zawierający MNIST/")
    parser.add_argument("--out", type=Path, default=Path("wyniki_analizy_mnist"))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps musi być dodatnie")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_model(args.weights, device)
    filters, optimization, examples = [args.out / p for p in
        ("01_maski_conv1", "02_maksymalizacja", "03_przyklady_MNIST")]
    for folder in (filters, optimization, examples):
        folder.mkdir(parents=True, exist_ok=True)
    plot_filters(model.conv1.weight[:, 0].cpu().numpy(), filters)
    for layer in ("conv1", "conv2"):
        for objective in OBJECTIVES:
            folder = optimization / objective
            folder.mkdir(exist_ok=True)
            results = maximize(model, layer, args.steps, device, objective)
            plot_optimization(layer, *results, folder, objective)
    dataset = MNIST(root=str(args.data), train=False, download=True)
    plot_examples(model, dataset, device, examples)
    print(f"Zapisano wyniki w {args.out.resolve()}")


if __name__ == "__main__":
    main()
