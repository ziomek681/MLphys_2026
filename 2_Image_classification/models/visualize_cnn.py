from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# ============================================================
# USTAWIENIA
# ============================================================
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "CNN_best.pth"
OUT_DIR = HERE / "visualizations"

DIGIT_TO_ANALYZE = 4
N_REAL_EXAMPLES = 4

FEATURE_STEPS = 250
FEATURE_LR = 0.05
FEATURE_RESTARTS = 3             # kilka niezależnych losowych początków
TV_WEIGHT = 0.01                 # regularyzacja wygładzająca

# Dokładnie taka normalizacja jak przy treningu MNIST
MEAN = 0.1307
STD = 0.3081


# ============================================================
# ARCHITEKTURA IDENTYCZNA JAK PODCZAS TRENINGU
# ============================================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(10 * 11 * 11, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2_drop(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def normalize(image):
    """Obraz w [0, 1] -> wejście modelu z normalizacją MNIST."""
    return (image - MEAN) / STD


def activation_maps(model, image, layer, after_relu=True):
    """Aktywacja conv1/conv2 przed lub po ReLU; image: (N, 1, 28, 28) w [0, 1]."""
    x = model.conv1(normalize(image))
    if layer == "conv1":
        return F.relu(x) if after_relu else x

    x = model.conv2(F.relu(x))
    if layer == "conv2":
        return F.relu(x) if after_relu else x

    raise ValueError("layer musi być 'conv1' lub 'conv2'")


def total_variation(image):
    return (
        (image[:, :, 1:, :] - image[:, :, :-1, :]).square().mean()
        + (image[:, :, :, 1:] - image[:, :, :, :-1]).square().mean()
    )


# ============================================================
# 1. FILTRY conv1: 3 WIERSZE x 2 KOLUMNY + OSOBNY COLORBAR
# ============================================================
def plot_conv1_filters(model):
    weights = model.conv1.weight.detach().cpu()[:, 0]
    vmax = max(weights.abs().max().item(), 1e-8)

    fig = plt.figure(figsize=(8, 9))
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.08],
                           left=0.08, right=0.92, top=0.92, bottom=0.06,
                           wspace=0.28, hspace=0.35)

    for i, weight in enumerate(weights):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        image = ax.imshow(weight, cmap="gray", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
        ax.set_title(f"Filtr {i}")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

    fig.add_subplot(grid[2, 1]).axis("off")
    cax = fig.add_subplot(grid[:, 2])  # colorbar ma osobną kolumnę
    fig.colorbar(image, cax=cax, label="Wartość wagi")

    fig.suptitle("Wagi filtrów conv1 (3×3)")
    fig.savefig(OUT_DIR / "01_filters_conv1_grayscale.png", dpi=200)
    plt.close(fig)


# ============================================================
# 2. FEATURE VISUALIZATION
# ============================================================
def optimize_channel(model, layer, channel):
    """
    Maksymalizujemy średnią odpowiedź PRZED końcowym ReLU wybranego kanału.
    Sprawdzamy także rzeczywistą aktywację PO ReLU.
    Wybieramy najlepszy wynik z kilku losowych punktów startowych.
    """
    best = None

    for _ in range(FEATURE_RESTARTS):
        image = torch.rand(1, 1, 28, 28, requires_grad=True)
        with torch.no_grad():
            initial_raw = activation_maps(model, image, layer, after_relu=False)[:, channel].mean().item()
            initial_relu = activation_maps(model, image, layer, after_relu=True)[:, channel].mean().item()

        optimizer = torch.optim.Adam([image], lr=FEATURE_LR)
        for _ in range(FEATURE_STEPS):
            optimizer.zero_grad()
            raw_score = activation_maps(model, image, layer, after_relu=False)[:, channel].mean()
            loss = -raw_score + TV_WEIGHT * total_variation(image)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                image.clamp_(0, 1)

        with torch.no_grad():
            raw_map = activation_maps(model, image, layer, after_relu=False)[:, channel]
            final_raw = raw_map.mean().item()
            final_relu = F.relu(raw_map).mean().item()
            final_raw_max = raw_map.max().item()

        candidate = {
            "image": image.detach().clone(),
            "initial_raw": initial_raw,
            "initial_relu": initial_relu,
            "final_raw": final_raw,
            "final_relu": final_relu,
            "final_raw_max": final_raw_max,
        }
        # Preferujemy faktyczną aktywację po ReLU; przy zerze lepszą preaktywację.
        if best is None or (final_relu, final_raw) > (best["final_relu"], best["final_raw"]):
            best = candidate

    return best


def plot_feature_visualization(model, layer):
    n_channels = getattr(model, layer).out_channels
    results = [optimize_channel(model, layer, ch) for ch in range(n_channels)]

    # Obrazy maksymalizujące kanały
    ncols = 5
    nrows = (n_channels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows), squeeze=False,
                             layout="constrained")
    for ch, result in enumerate(results):
        ax = axes.flat[ch]
        ax.imshow(result["image"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Kanał {ch}\nReLU: {result['initial_relu']:.2f} → {result['final_relu']:.2f}", fontsize=11)
        ax.axis("off")
    for ax in list(axes.flat)[n_channels:]:
        ax.axis("off")
    fig.suptitle(f"Feature visualization: {layer} (cel: aktywacja przed ReLU)")
    fig.savefig(OUT_DIR / f"02_feature_visualization_{layer}.png", dpi=200)
    plt.close(fig)

    # Test: czy wzrosła preaktywacja oraz faktyczna aktywacja po ReLU?
    channels = list(range(n_channels))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout="constrained")

    for ax, before, after, title, ylabel in [
        (axes[0], [r["initial_raw"] for r in results],
         [r["final_raw"] for r in results],
         "Optymalizowany cel (przed ReLU)", "Średnia preaktywacja"),
        (axes[1], [r["initial_relu"] for r in results],
         [r["final_relu"] for r in results],
         "Rzeczywista odpowiedź (po ReLU)", "Średnia aktywacja"),
    ]:
        ax.plot(channels, before, "o-", label="przed")
        ax.plot(channels, after, "o-", label="po")
        ax.set_title(title)
        ax.set_xlabel("Kanał docelowy")
        ax.set_ylabel(ylabel)
        ax.set_xticks(channels)
        ax.grid(alpha=0.3)
        ax.legend()

    # Macierz: jaki kanał odpowiada na obraz zoptymalizowany pod dany kanał?
    score_matrix = torch.zeros(n_channels, n_channels)
    with torch.no_grad():
        for ch, result in enumerate(results):
            maps = activation_maps(model, result["image"], layer)[0]
            score_matrix[ch] = maps.mean(dim=(1, 2))

    im = axes[2].imshow(score_matrix.numpy(), cmap="magma", interpolation="nearest")
    axes[2].set_title("Aktywacja wszystkich kanałów")
    axes[2].set_xlabel("Kanał zmierzony")
    axes[2].set_ylabel("Kanał optymalizowany")
    axes[2].set_xticks(channels)
    axes[2].set_yticks(channels)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(OUT_DIR / f"03_feature_visualization_check_{layer}.png", dpi=200)
    plt.close(fig)

    print(f"\nKontrola feature visualization — {layer}")
    for ch, r in enumerate(results):
        status = "AKTYWOWANY" if r["final_relu"] > 1e-6 else "NADAL ZEROWY"
        print(f"  kanał {ch:2}: pre-ReLU {r['initial_raw']:+.3f} -> {r['final_raw']:+.3f}, "
              f"po ReLU {r['initial_relu']:.3f} -> {r['final_relu']:.3f}, "
              f"maks. pre-ReLU {r['final_raw_max']:+.3f}  [{status}]")
    print("  Uwaga: aktywacja innych kanałów nie oznacza błędu optymalizacji.")


# ============================================================
# 3. AKTYWACJE DLA KILKU PRAWDZIWYCH PRZYKŁADÓW
# ============================================================
def collect_examples(dataset, digit, count):
    examples = []
    for image, label in dataset:
        if label == digit:
            examples.append((image, label))
            if len(examples) == count:
                break
    if len(examples) != count:
        raise RuntimeError(f"Znaleziono tylko {len(examples)} przykładów cyfry {digit}.")
    return examples


def plot_real_examples(model, examples, layer):
    n_examples = len(examples)
    n_channels = getattr(model, layer).out_channels
    fig, axes = plt.subplots(n_examples, n_channels + 1,
                             figsize=(2.0 * (n_channels + 1), 2.25 * n_examples),
                             squeeze=False, layout="constrained")

    with torch.no_grad():
        all_maps = [activation_maps(model, image.unsqueeze(0), layer)[0].cpu()
                    for image, _ in examples]
    vmax = max((m.max().item() for m in all_maps), default=1.0)
    vmax = max(vmax, 1e-8)

    for row, ((image, label), maps) in enumerate(zip(examples, all_maps)):
        with torch.no_grad():
            pred = model(normalize(image.unsqueeze(0))).argmax(dim=1).item()
        axes[row, 0].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Przykład {row + 1}\ntrue={label}, pred={pred}", fontsize=11)
        axes[row, 0].axis("off")

        for ch in range(n_channels):
            im = axes[row, ch + 1].imshow(maps[ch], cmap="magma", vmin=0, vmax=vmax,
                                          interpolation="nearest")
            if row == 0:
                axes[row, ch + 1].set_title(f"Kanał {ch}")
            axes[row, ch + 1].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01,
                 label="Aktywacja po ReLU")
    fig.suptitle(f"Aktywacje {layer} dla {n_examples} przykładów cyfry {DIGIT_TO_ANALYZE}")
    fig.savefig(OUT_DIR / f"04_real_examples_{layer}_digit{DIGIT_TO_ANALYZE}.png", dpi=200)
    plt.close(fig)


def check_channels_on_mnist(model, dataset, count=256):
    """Czy kanały aktywują się na prawdziwych obrazach, nie tylko syntetycznych?"""
    count = min(count, len(dataset))
    images = torch.stack([dataset[i][0] for i in range(count)])
    print(f"\nAktywność kanałów na pierwszych {count} obrazach testowych MNIST:")
    with torch.no_grad():
        for layer in ("conv1", "conv2"):
            maps = activation_maps(model, images, layer, after_relu=True)
            active_fraction = (maps > 0).float().mean(dim=(0, 2, 3))
            print(f"  {layer} — odsetek aktywnych pikseli na mapach:")
            for channel, fraction in enumerate(active_fraction):
                print(f"    kanał {channel:2}: {fraction.item() * 100:.1f}%")
    print("  0% w tej próbce: kanał nie zareagował na żaden badany obraz,")
    print("  ale nie dowodzi to, że nie mógłby zareagować na inny obraz.")


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(123)

    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    model.requires_grad_(False)

    candidates = [HERE / "data", HERE.parent / "data", HERE.parent.parent / "data"]
    data_dir = next((p for p in candidates if (p / "MNIST").exists()), candidates[0])
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True,
                             transform=transforms.ToTensor())
    examples = collect_examples(test_set, DIGIT_TO_ANALYZE, N_REAL_EXAMPLES)
    check_channels_on_mnist(model, test_set)

    plot_conv1_filters(model)
    for layer in ("conv1", "conv2"):
        plot_feature_visualization(model, layer)
        plot_real_examples(model, examples, layer)

    print(f"\nWyniki zapisane w: {OUT_DIR}")


if __name__ == "__main__":
    main()
