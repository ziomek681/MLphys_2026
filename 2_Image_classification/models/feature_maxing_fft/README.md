# Maksymalizacja kanałów w domenie Fouriera — MNIST

Architektura z `models/visualize_cnn.py`, wagi z `CNN_best.pth`. Objęto wszystkie 5 kanałów conv1 (splot 3×3, krok 2) i 10 kanałów conv2 (3×3, krok 1). Każdy PNG ma 28×28 pikseli. Przed splotem model otrzymał (x − 0,1307) / 0,3081.

Każdy obraz powstał osobno dla jednego kanału (numeracja od zera). Celem była średnia przestrzenna wyjścia wskazanego splotu przed jego ReLU. Wcześniejsze warstwy działają jak podczas predykcji. Wysoki wynik oznacza silną odpowiedź kanału na wzór w wielu pozycjach. Jednolite pola mogą wzmacniać filtry wrażliwe na jasność.

Adam (180 kroków, lr 0,09) zmieniał rzeczywiste i urojone współczynniki widma. Przed odwrotną FFT każdy współczynnik pomnożono przez 0,25 / max(f, 1/N), gdzie f jest częstotliwością radialną w cyklach/piksel, a N bokiem optymalizowanego obrazu lub kafla. Sigmoid ograniczał jasność do [0,1]. Wybrano najlepszy z trzech losowych startów według końcowej wartości celu. [Olah i in., Feature Visualization](https://distill.pub/2017/feature-visualization/)

Atlasy pokazują zestawienie warstw. Kliknięcie numeru kanału w tabeli otwiera oryginalny PNG. Obie kolumny mierzą średnią przed ReLU na pełnym wejściu modelu dla wybranej próby. Wynik końcowy przeliczono po odczycie 8-bitowego PNG.

| Warstwa | Kanał | Start | PNG |
|---|---:|---:|---:|
| conv1 | [0](conv1_k000.png) | -2.730 | 0.575 |
| conv1 | [1](conv1_k001.png) | -1.596 | 0.426 |
| conv1 | [2](conv1_k002.png) | 2.107 | 4.344 |
| conv1 | [3](conv1_k003.png) | 3.308 | 6.438 |
| conv1 | [4](conv1_k004.png) | 0.097 | 1.103 |
| conv2 | [0](conv2_k000.png) | -3.755 | 0.379 |
| conv2 | [1](conv2_k001.png) | 1.288 | 3.212 |
| conv2 | [2](conv2_k002.png) | -5.032 | 0.821 |
| conv2 | [3](conv2_k003.png) | -4.363 | 0.493 |
| conv2 | [4](conv2_k004.png) | -2.902 | 0.424 |
| conv2 | [5](conv2_k005.png) | -0.636 | 0.527 |
| conv2 | [6](conv2_k006.png) | -0.876 | 0.596 |
| conv2 | [7](conv2_k007.png) | -5.994 | 0.223 |
| conv2 | [8](conv2_k008.png) | -3.351 | 0.318 |
| conv2 | [9](conv2_k009.png) | -5.751 | 0.630 |
