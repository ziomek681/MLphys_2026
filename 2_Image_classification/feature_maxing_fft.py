import ast
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.manual_seed(2026)


def load_model(source, weights, **kwargs):
    # Oba pliki treningowe mają wykonywalny kod na poziomie modułu.
    tree = ast.parse((ROOT / source).read_text(encoding="utf-8-sig"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CNN")
    scope = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module([definition], []), source, "exec"), scope)
    model = scope["CNN"](**kwargs)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.eval().requires_grad_(False)


def select_channels(net):
    y, x = torch.meshgrid(torch.arange(40), torch.arange(40), indexing="ij")
    probes = []
    for kx, ky in [(1,0),(0,1),(1,1),(1,-1),(2,1),(1,2),(2,-1),(1,-2)]:
        for freq in (1, 2, 4, 7):
            phase = 2 * math.pi * freq * (kx*x + ky*y) / 40
            probes.extend([phase.cos(), (phase.cos() + (phase.T + 0.7).cos())/2])
    maps = F.pad((torch.stack(probes)[:, None]+1)/2, (12,)*4, mode="circular")
    chosen = {}
    seeds = {}
    for i, module in enumerate(net):
        maps = module(maps)
        if isinstance(module, nn.Conv2d):
            margin = 12 // (2 ** (i//3))
            response = maps[:, :, margin:-margin, margin:-margin].relu().mean((2,3)).T
            centered = response - response.mean(1, keepdim=True)
            strength = centered.norm(dim=1)
            profile = F.normalize(centered)
            eligible = strength >= strength[strength > 1e-6].median()
            ids = [int(strength.argmax())]
            while len(ids) < 6:
                distance = 1 - (profile @ profile[ids].T).amax(1)
                distance[~eligible] = -10
                distance[ids] = -10
                ids.append(int(distance.argmax()))
            chosen[i] = sorted(ids)
            seeds[i] = (torch.stack(probes)[response[chosen[i]].argmax(1)][:, None] + 1) / 2
    return chosen, seeds


def synthesize(crystal):
    if crystal:
        path = ROOT / "ResultsCrystal/8/CNN_checkpoint_8"
        model = load_model("Crystal.py", path, input_size=500, kernel_size=3, CNN_masks=[16,64,128], CNN_series=[1,1,1])
        net = list(model.CNN)[:7]
        targets, seeds = select_channels(net)
        tile, size = 40, 500
    else:
        path = ROOT / "models/CNN_best.pth"
        model = load_model("models/visualize_cnn.py", path)
        net = [model.conv1, nn.ReLU(), model.conv2]
        targets = {0:list(range(5)), 2:list(range(10))}
        tile = size = 28
    out = path.parent / "feature_maxing_fft_po_relu"
    out.mkdir(exist_ok=True)
    report = []
    for layer, channels in targets.items():
        name = f"CNN.{layer}" if crystal else f"conv{1+layer//2}"
        count = len(channels)
        rows = torch.arange(count)
        prefix = nn.Sequential(*net[:layer+1])
        def scores(image, patch):
            if crystal and patch:
                image = F.pad(image, (12,)*4, mode="circular")
            maps = prefix(image if crystal else (image-.1307)/.3081)
            selected = maps[rows, channels]
            if crystal and patch:
                m = 12 // (2 ** (layer//3))
                selected = selected[:, m:-m, m:-m]
            return selected.relu().mean((1,2))
        def expand(image):
            repeats = math.ceil(size/tile)
            return image.repeat(1,1,repeats,repeats)[:,:,:size,:size]
        radius = torch.hypot(torch.fft.fftfreq(tile)[:,None], torch.fft.rfftfreq(tile))
        scale = .25 / radius.clamp_min(1/tile)
        best = torch.full((count,), -torch.inf)
        winners = torch.zeros(count,1,tile,tile)
        starts = winners.clone()
        for restart in range(3):
            coeff = (torch.randn(count,1,tile,tile//2+1,2)*1.5).requires_grad_()
            if crystal and restart == 2:
                logits = torch.logit(seeds[layer].clamp(.005, .995))
                seed_fft = torch.fft.rfft2(logits, norm="ortho") / scale
                coeff = torch.view_as_real(seed_fft).contiguous().requires_grad_()
            def decode():
                return torch.fft.irfft2(torch.view_as_complex(coeff)*scale, s=(tile,tile), norm="ortho").sigmoid()
            initial = decode().detach()
            optimizer = torch.optim.Adam([coeff], lr=.09)
            for step in range(180):
                optimizer.zero_grad()
                (-scores(decode(), True).sum()).backward()
                optimizer.step()
            with torch.no_grad():
                final = decode()
                result = scores(final, True)
                better = result > best
                winners[better] = final[better]
                starts[better] = initial[better]
                best = torch.maximum(best,result)
        images = (expand(winners)*255).round().byte()
        canvas = Image.new("RGB",(min(count,5)*150, math.ceil(count/5)*166),"#f6f5f2")
        draw = ImageDraw.Draw(canvas)
        for n, channel in enumerate(channels):
            stem = f"{name}_k{channel:03d}.png"
            png = Image.fromarray(images[n,0].numpy())
            png.save(out/stem)
            x,y = n%5*150,n//5*166
            canvas.paste(png.resize((140,140),Image.Resampling.NEAREST if not crystal else Image.Resampling.LANCZOS),(x+5,y))
            draw.text((x+5,y+144),f"{name} / {channel}",fill="black")
        canvas.save(out/f"atlas_{name}.png")
        # Zweryfikuj odpowiedź na bajtach odczytanych z docelowych PNG.
        decoded = torch.stack([torch.from_numpy(np.array(Image.open(out/f"{name}_k{ch:03d}.png"))) for ch in channels])[:,None]/255
        with torch.no_grad():
            before = scores(expand(starts),False)
            after = scores(decoded,False)
        assert torch.all(after > before), (channels, before, after)
        report.extend((name,ch,float(a),float(b)) for ch,a,b in zip(channels,before,after))
        print(name, channels, flush=True)

    title = "Wallpaper / Crystal 8" if crystal else "MNIST"
    details = ("Architektura z `Crystal.py`, wagi z `CNN_checkpoint_8`. Trzy sploty 3×3 mają 16, 64 i 128 kanałów. "
               "Wybrano po 6 na warstwę. Każdy kanał oceniono na 64 sondach sinusoidalnych (8 wektorów kierunku, 4 częstotliwości, pojedyncza lub złożona fala). "
               "Spośród kanałów reagujących na sondy, ze zmiennością nie mniejszą niż mediana w tej grupie, wybrano najpierw najbardziej zmienny, potem kolejne o najmniejszym maksymalnym podobieństwie kosinusowym do już wybranych profili po odjęciu średniej. "
               "Optymalizowano okresowy kafel 40×40 z cyklicznym marginesem 12 px, mierząc odpowiedź w jego wnętrzu. PNG powtarza kafel do 500×500, w skali jasności wejścia sieci po treningowej inwersji. Dwa starty były losowe, a trzeci wykorzystał sondę o największej odpowiedzi danego kanału po ReLU. Okres 40 px wynika z konstrukcji eksperymentu."
               if crystal else "Architektura z `models/visualize_cnn.py`, wagi z `CNN_best.pth`. Objęto wszystkie 5 kanałów conv1 (splot 3×3, krok 2) i 10 kanałów conv2 (3×3, krok 1). Każdy PNG ma 28×28 pikseli. Przed splotem model otrzymał (x − 0,1307) / 0,3081.")
    body = f"# Maksymalizacja kanałów w domenie Fouriera — {title}\n\n{details}\n\n"
    body += "Każdy obraz powstał osobno dla jednego kanału (numeracja od zera). Celem była średnia przestrzenna wyjścia wskazanego splotu po zastosowaniu ReLU do każdej pozycji mapy. Wcześniejsze warstwy działają jak podczas predykcji. Wysoki wynik oznacza silną odpowiedź kanału na wzór w wielu pozycjach. Jednolite pola mogą wzmacniać filtry wrażliwe na jasność.\n\n"
    body += "Adam (180 kroków, lr 0,09) zmieniał rzeczywiste i urojone współczynniki widma. Przed odwrotną FFT każdy współczynnik pomnożono przez 0,25 / max(f, 1/N), gdzie f jest częstotliwością radialną w cyklach/piksel, a N bokiem optymalizowanego obrazu lub kafla. Sigmoid ograniczał jasność do [0,1]. Wybrano najlepszy z trzech startów według końcowej wartości celu. [Olah i in., Feature Visualization](https://distill.pub/2017/feature-visualization/)\n\n"
    body += "Atlasy pokazują zestawienie warstw. Kliknięcie numeru kanału w tabeli otwiera oryginalny PNG. Obie kolumny mierzą średnią po ReLU na pełnym wejściu modelu dla wybranej próby. Wynik końcowy przeliczono po odczycie 8-bitowego PNG.\n\n| Warstwa | Kanał | Start | PNG |\n|---|---:|---:|---:|\n"
    for name,ch,a,b in report:
        body += f"| {name} | [{ch}]({name}_k{ch:03d}.png) | {a:.3f} | {b:.3f} |\n"
    (out/"README.md").write_text(body, encoding="utf-8")

if __name__ == "__main__":
    synthesize(False)
    import runpy
    runpy.run_path(str(ROOT / "ResultsCrystal/8/feature_maxing_fft_500/analiza.py"), run_name="__main__")
