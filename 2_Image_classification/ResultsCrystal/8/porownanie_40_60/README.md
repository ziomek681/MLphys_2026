# Crystal 8 — porównanie kafli 40 i 60 px

Te same kanały i wagi. Bazą są zapisane PNG z eksperymentu 40 px po ReLU. Dla 60 px wykonano ponownie 180 kroków Adama (lr 0,09) z trzech startów w domenie Fouriera. Dwa były losowe, trzeci wybrano spośród 64 sond generowanych na nowej siatce. Zmiana rozmiaru zmienia również dostępne częstotliwości sond i realizację losową. Obie aktywacje zmierzono na pełnych obrazach 500×500 po kwantyzacji do PNG.

Atlas pokazuje wycinki 120×120 px w tej samej skali. Okres i kąt linii podano dla najsilniejszej niezerowej składowej FFT pojedynczego kafla. Udział tej pary częstotliwości w mocy po odjęciu średniej pomaga ocenić, czy przypisanie jednego okresu ma sens.

| Splot | Kanał | Akt. 40 | Akt. 60 | Zmiana | Okres 40 / 60 [px] | Kąt 40 / 60 [°] | Udział 40 / 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN.0 | 3 | 0.2125 | 0.2248 | +5.8% | 2.6 / 3.8 | 63 / 63 | 76% / 78% |
| CNN.0 | 4 | 0.2034 | 0.1700 | -16.4% | 4.5 / 6.1 | 117 / 135 | 75% / 71% |
| CNN.0 | 6 | 0.2080 | 0.2084 | +0.2% | 4.0 / 3.8 | 174 / 153 | 1% / 74% |
| CNN.0 | 13 | 0.1932 | 0.2097 | +8.6% | 2.6 / 3.8 | 117 / 117 | 66% / 66% |
| CNN.0 | 14 | 0.3258 | 0.3452 | +6.0% | 2.6 / 60.0 | 117 / 0 | 30% / 12% |
| CNN.0 | 15 | 0.1295 | 0.1255 | -3.1% | 2.6 / 60.0 | 63 / 0 | 69% / 2% |
| CNN.3 | 14 | 1.1087 | 0.9895 | -10.7% | 7.1 / 60.0 | 45 / 90 | 39% / 16% |
| CNN.3 | 28 | 3.4175 | 3.3357 | -2.4% | 2.0 / 2.1 | 117 / 72 | 1% / 0% |
| CNN.3 | 31 | 1.2145 | 0.9147 | -24.7% | 2.6 / 9.4 | 117 / 141 | 69% / 2% |
| CNN.3 | 43 | 1.3215 | 1.2590 | -4.7% | 2.6 / 60.0 | 153 / 0 | 75% / 1% |
| CNN.3 | 47 | 0.5041 | 0.5076 | +0.7% | 1.5 / 1.4 | 45 / 46 | 6% / 4% |
| CNN.3 | 52 | 1.9087 | 1.3918 | -27.1% | 2.6 / 11.8 | 117 / 11 | 65% / 1% |
| CNN.6 | 1 | 1.4191 | 1.4529 | +2.4% | 17.9 / 11.8 | 117 / 79 | 13% / 2% |
| CNN.6 | 27 | 25.6415 | 23.1220 | -9.8% | 2.6 / 2.2 | 27 / 121 | 44% / 0% |
| CNN.6 | 45 | 2.1113 | 1.9322 | -8.5% | 8.9 / 13.4 | 63 / 63 | 36% / 58% |
| CNN.6 | 97 | 4.2133 | 3.8395 | -8.9% | 8.0 / 3.8 | 90 / 63 | 7% / 37% |
| CNN.6 | 106 | 3.3645 | 3.3863 | +0.6% | 17.9 / 30.0 | 117 / 0 | 6% / 5% |
| CNN.6 | 124 | 3.2091 | 3.1744 | -1.1% | 28.3 / 60.0 | 45 / 90 | 9% / 8% |
