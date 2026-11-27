from pathlib import Path
import sys
import math
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

out = Path(__file__).resolve().parent
sys.path.insert(0, str(out.parents[2]))
from feature_maxing_fft import load_model, select_channels

model = load_model("Crystal.py", out.parent/"CNN_checkpoint_8", input_size=500,
                   kernel_size=3, CNN_masks=[16,64,128], CNN_series=[1,1,1])
layers = list(model.CNN)[:7]
targets, _ = select_channels(layers)
n = 60
y,x = torch.meshgrid(torch.arange(n),torch.arange(n),indexing="ij")
probes = []
for kx,ky in [(1,0),(0,1),(1,1),(1,-1),(2,1),(1,2),(2,-1),(1,-2)]:
    for k in (1,2,4,7):
        a = (x*kx+y*ky)*2*math.pi*k/n
        probes.extend([a.cos(), (a.cos()+(a.T+.7).cos())/2])
probes = (torch.stack(probes)[:,None]+1)/2
scale = .25/torch.hypot(torch.fft.fftfreq(n)[:,None],torch.fft.rfftfreq(n)).clamp_min(1/n)
atlas = Image.new("RGB",(6*138,3*288),"white")
draw = ImageDraw.Draw(atlas)
report = ["# Crystal 8 — porównanie kafli 40 i 60 px","",
"Te same kanały i wagi. Bazą są zapisane PNG z eksperymentu 40 px po ReLU. Dla 60 px wykonano ponownie 180 kroków Adama (lr 0,09) z trzech startów w domenie Fouriera. Dwa były losowe, trzeci wybrano spośród 64 sond generowanych na nowej siatce. Zmiana rozmiaru zmienia również dostępne częstotliwości sond i realizację losową. Obie aktywacje zmierzono na pełnych obrazach 500×500 po kwantyzacji do PNG.",
"", "Atlas pokazuje wycinki 120×120 px w tej samej skali. Okres i kąt linii podano dla najsilniejszej niezerowej składowej FFT pojedynczego kafla. Udział tej pary częstotliwości w mocy po odjęciu średniej pomaga ocenić, czy przypisanie jednego okresu ma sens.", "",
"| Splot | Kanał | Akt. 40 | Akt. 60 | Zmiana | Okres 40 / 60 [px] | Kąt 40 / 60 [°] | Udział 40 / 60 |",
"|---|---:|---:|---:|---:|---:|---:|---:|"]
def peak(im, size):
    t = np.asarray(im)[:size,:size].astype(float)
    power = abs(np.fft.fft2(t-t.mean()))**2
    i,j = np.unravel_index(power.argmax(),power.shape)
    v,u = np.fft.fftfreq(size)[[i,j]]
    if power.sum()==0:
        return np.nan,np.nan,0
    return 1/np.hypot(u,v),(np.degrees(np.arctan2(v,u))+90)%180,(power[i,j]+(0 if (i,j)==((-i)%size,(-j)%size) else power[-i,-j]))/power.sum()
for row,(idx,ids) in enumerate(targets.items()):
    prefix = torch.nn.Sequential(*layers[:idx+1])
    margin = 12//(2**row)
    def patch_scores(t):
        m = prefix(F.pad(t,(12,)*4,mode="circular")).relu()
        return m[:,:,margin:-margin,margin:-margin].mean((2,3))
    r = torch.arange(len(ids))
    chosen = patch_scores(probes)[:,ids].argmax(0)
    best = torch.full((len(ids),),-torch.inf)
    tiles = torch.zeros(len(ids),1,n,n)
    for start in range(3):
        c = torch.randn(len(ids),1,n,n//2+1,2)*1.5
        if start==2:
            c = torch.view_as_real(torch.fft.rfft2(torch.logit(probes[chosen].clamp(.005,.995)),norm="ortho")/scale).contiguous()
        c.requires_grad_()
        opt = torch.optim.Adam([c],lr=.09)
        def decode():
            return torch.fft.irfft2(torch.view_as_complex(c)*scale,s=(n,n),norm="ortho").sigmoid()
        for _ in range(180):
            opt.zero_grad()
            (-patch_scores(decode())[r,ids].sum()).backward()
            opt.step()
        with torch.no_grad():
            t = decode()
            s = patch_scores(t)[r,ids]
            tiles[s>best] = t[s>best]
            best = torch.maximum(s,best)
    for col,ch in enumerate(ids):
        file = f"CNN.{idx}_k{ch:03d}.png"
        new = (tiles[col,0].repeat(9,9)[:500,:500]*255).round().byte().numpy()
        Image.fromarray(new).save(out/file)
        ims = [Image.open(out.parent/"feature_maxing_fft_po_relu"/file),Image.open(out/file)]
        scores = []
        for j,im in enumerate(ims):
            t = torch.tensor(np.array(im)).float()[None,None]/255
            scores.append(prefix(t)[0,ch].relu().mean().item())
            xx,yy = col*138,row*288+j*139
            atlas.paste(im.crop((0,0,120,120)),(xx,yy+16))
            draw.text((xx,yy),f"CNN.{idx}/{ch}   {40+20*j}px",fill="black")
        a,b = scores
        p,q = [peak(im,size) for im,size in zip(ims,[40,60])]
        report.append(f"| CNN.{idx} | {ch} | {a:.4f} | {b:.4f} | {100*(b/a-1):+.1f}% | {p[0]:.1f} / {q[0]:.1f} | {p[1]:.0f} / {q[1]:.0f} | {p[2]:.0%} / {q[2]:.0%} |")
    print(*report[-6:],sep="\n",flush=True)
atlas.save(out/"atlas.png")
(out/"README.md").write_text("\n".join(report)+"\n",encoding="utf-8")
