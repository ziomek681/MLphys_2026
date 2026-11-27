"""Pełnowymiarowa maksymalizacja kanałów Crystal 8 po ReLU."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[2]))
from feature_maxing_fft import load_model

SIZE = 500
STEPS = 120
CHANNELS = {0: [3, 4, 6, 13, 14, 15], 3: [14, 28, 31, 43, 47, 52],
            6: [1, 27, 45, 97, 106, 124]}
# Wyniki referencyjne po ReLU z porównania 40/60, pomiar na pełnym PNG.
BASE60 = {0: [.2248, .1700, .2084, .2097, .3452, .1255],
          3: [.9895, 3.3357, .9147, 1.2590, .5076, 1.3918],
          6: [1.4529, 23.1220, 1.9322, 3.8395, 3.3863, 3.1744]}


def run():
    model = load_model("Crystal.py", OUT.parent / "CNN_checkpoint_8", input_size=SIZE,
                       kernel_size=3, CNN_masks=[16, 64, 128], CNN_series=[1, 1, 1])
    freq = torch.hypot(torch.fft.fftfreq(SIZE)[:, None], torch.fft.rfftfreq(SIZE))
    scale = .25 / freq.clamp_min(1 / SIZE)
    report = ["# Crystal 8 — maksymalizacja po ReLU na pełnym obrazie", "",
              "Każdy PNG ma 500×500 px, tyle co wejście treningowe. Dla każdego kanału optymalizowano osobne widmo całego obrazu. Sieć w każdym kroku otrzymywała wszystkie piksele. Każdy atlas pokazuje centralne wycinki 160×160 px w skali 1:1. Kliknięcie numeru kanału otwiera cały obraz.", "",
              "Użyto wag z `CNN_checkpoint_8` i architektury z `Crystal.py`. Zachowano 18 kanałów z porównania kafli 40 i 60 px. Cel to średnia po wszystkich pozycjach mapy po ReLU, przed następnym poolingiem. Adam (120 kroków, lr 0,09) zmieniał zespolone współczynniki Fouriera. Wagi widma wynosiły 0,25 / max(f, 1/500), gdzie f to częstotliwość radialna w cyklach/piksel. Po odwrotnej FFT sigmoid ograniczał jasność do zakresu [0,1]. Każdy kanał miał osobny losowy start. Przy wyniku poniżej połowy wcześniejszej aktywacji z kafla 60 px dodano drugi start z fali o zmiennej fazie i szumem, wybranej na podstawie odpowiedzi na sondy kierunkowe. Zachowano silniejszy wynik.", "",
              "Pola recepcyjne pojedynczych pozycji po kolejnych splotach mają boki 3, 8 i 18 px. Drobna struktura może zatem pojawiać się w wielu miejscach także przy swobodnej optymalizacji całego obrazu. Start i końcowy PNG oceniono na pełnym wejściu. Kolumna 60 px to wcześniejszy pomiar tego samego kanału na powielonym kaflu, przy innym budżecie optymalizacji.", ""]
    for layer, ids in CHANNELS.items():
        prefix = torch.nn.Sequential(*list(model.CNN)[:layer])
        conv = model.CNN[layer]
        atlas = Image.new("L", (3 * 172, 2 * 191), 245)
        draw = ImageDraw.Draw(atlas)
        report.extend([f"## CNN.{layer}", "", f"![Detale kanałów](atlas_CNN.{layer}.png)", "",
                       "| Kanał | Start | PNG 500 px | PNG z kafla 60 px |", "|---|---:|---:|---:|"])
        for i, channel in enumerate(ids):
            def score(image):
                maps = F.conv2d(prefix(image), conv.weight[channel:channel+1],
                                conv.bias[channel:channel+1], padding=conv.padding)
                return maps.relu().mean()
            torch.manual_seed(2026 + 1000 * layer + channel)
            def optimize(coeff):
                coeff = coeff.contiguous().requires_grad_()
                optimizer = torch.optim.Adam([coeff], lr=.09)
                def decode():
                    spectrum = torch.view_as_complex(coeff) * scale
                    return torch.fft.irfft2(spectrum, s=(SIZE,SIZE), norm="ortho").sigmoid()
                for step in range(STEPS):
                    optimizer.zero_grad()
                    activation = score(decode())
                    if step == 0:
                        initial = activation.item()
                    (-activation).backward()
                    optimizer.step()
                with torch.no_grad():
                    result = decode()
                    return initial, result, score(result).item()

            initial, result, achieved = optimize(torch.randn(1, 1, SIZE, SIZE//2+1, 2))
            if achieved < .5 * BASE60[layer][i]:
                # Wybierz kierunek i skalę na małych sondach, potem wygeneruj
                # fazę od nowa na pełnej siatce z łagodnym odkształceniem.
                def wave(side, kx, ky, freq, crossed, warped):
                    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
                    if warped:
                        x = x + 1.8 * (y/38 + x/53).sin()
                        y = y + 2.2 * (x/67 - y/43).cos()
                    phase = (x*kx+y*ky) * (2*np.pi*freq/60)
                    w = (phase.cos() + (phase.T+.7).cos())/2 if crossed else phase.cos()
                    return (.5 + .495*w)[None,None]
                candidates = [(a,b,k,c) for a,b in ((1,0),(0,1),(1,1),(1,-1),(2,1),(1,2),(2,-1),(1,-2))
                              for k in (1,2,4,7) for c in (False,True)]
                best = max(candidates, key=lambda args: score(wave(96,*args,False)).item())
                logits = torch.logit(wave(SIZE,*best,True)) + .1*torch.randn(1,1,SIZE,SIZE)
                coeff = torch.view_as_real(torch.fft.rfft2(logits,norm="ortho") / scale)
                other_start, other_image, other_score = optimize(coeff)
                if other_score > achieved:
                    initial, result = other_start, other_image
            name = f"CNN.{layer}_k{channel:03d}.png"
            with torch.no_grad():
                Image.fromarray((result[0,0]*255).round().byte().numpy()).save(OUT/name)
                with Image.open(OUT/name) as png:
                    pixels = np.array(png)
                    assert png.size == (SIZE,SIZE)
                    for shift in (40,60):
                        assert not np.array_equal(pixels[shift:],pixels[:-shift])
                        assert not np.array_equal(pixels[:,shift:],pixels[:,:-shift])
                    final = conv(prefix(torch.from_numpy(pixels)[None,None].float()/255))[0,channel].relu().mean().item()
                    x, y = i%3*172+6, i//3*191+5
                    atlas.paste(png.crop((170,170,330,330)), (x,y))
                    draw.text((x,y+164),f"CNN.{layer} / {channel}",fill=0)
            assert final > initial, (layer, channel, final, initial)
            report.append(f"| [{channel}]({name}) | {initial:.4f} | {final:.4f} | {BASE60[layer][i]:.4f} |")
            print(layer, channel, initial, final, flush=True)
        atlas.save(OUT/f"atlas_CNN.{layer}.png")
        report.append("")
    (OUT/"README.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    run()
