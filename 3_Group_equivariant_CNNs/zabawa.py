import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

def translate(x, v):
    return x.roll((v[0],v[1]), dims=(-2, -1))

# A 90-deg rotation operation, where r is the number of rotations applied (i.e., r = 0, 1, 2, 3) indexing the $C_4$ group elements.
def rotate(x, r):
    return x.rot90(r, dims=(-2, -1))

x = torch.randn(1, 1, 9, 9)**2

r = 1
gx = rotate(x, r)

plt.imshow(x[0, 0].numpy())
plt.title('Original Image $x$')
plt.show()

plt.imshow(gx[0, 0].numpy())
plt.title('Rotated Image $g.x$')
plt.show()

filter3x3 = torch.randn(1, 1, 3, 3)

plt.imshow(filter3x3[0, 0].numpy())
plt.title('Filter')
plt.show()

# generate some input...
xs = torch.linspace(-5, 5, steps=16)
xg, yg = torch.meshgrid(xs, xs, indexing='xy')
x = torch.sin(torch.sqrt(xg * xg + yg * yg))
x = x[None, None, ...]

v = (2,3)  # translate by vector v=(2,3) (periodic b.c.)
gx = translate(x, v)

plt.imshow(x[0, 0].numpy())
plt.title('Original Image $x$')
plt.show()

plt.imshow(gx[0, 0].numpy())
plt.title('Translated Image $g.x$')
plt.show()

# ...then test equivariance:
psi_x = torch.conv2d(x, filter3x3, bias=None, padding=1)
#psi_x = torch.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), filter3x3, bias=None, padding=0)
psi_gx = torch.conv2d(gx, filter3x3, bias=None, padding=1)
#psi_gx = torch.conv2d(F.pad(gx, (1, 1, 1, 1), mode="circular"), filter3x3, bias=None, padding=0)

g_psi_x = translate(psi_x, v)

plt.imshow(g_psi_x[0, 0].numpy())
plt.title('$g.\\psi(x)$')
plt.show()

plt.imshow(psi_gx[0, 0].numpy())
plt.title('$\\psi(g.x)$')
plt.show()

plt.imshow((g_psi_x-psi_gx)[0, 0].numpy())
plt.title('$g.\\psi(x)-\\psi(g.x)$')
plt.show()

filter3x3 = torch.randn(1, 1, 3, 3)

plt.imshow(filter3x3[0, 0].numpy())
plt.title('Filter')
plt.show()

# generate some input...
xs = torch.linspace(-5, 5, steps=16)
xg, yg = torch.meshgrid(xs, xs, indexing='xy')
x = torch.sin(torch.sqrt(xg * xg + yg/3. * yg/3.))
x = x[None, None, ...]

r = 1
gx = rotate(x, r)

plt.imshow(x[0, 0].numpy())
plt.title('Original Image $x$')
plt.show()

plt.imshow(gx[0, 0].numpy())
plt.title('Rotated Image $g.x$')
plt.show()

# ...then test equivariance:
psi_x = torch.conv2d(x, filter3x3, bias=None, padding=1)
psi_gx = torch.conv2d(gx, filter3x3, bias=None, padding=1)

g_psi_x = rotate(psi_x, r)

plt.imshow(g_psi_x[0, 0].numpy())
plt.title('$g.\\psi(x)$')
plt.show()

plt.imshow(psi_gx[0, 0].numpy())
plt.title('$\\psi(g.x)$')
plt.show()

plt.imshow((g_psi_x-psi_gx)[0, 0].numpy())
plt.title('$g.\\psi(x)-\\psi(g.x)$')
plt.show()

filter3x3 = torch.empty((1, 1, 3, 3))
# fill the central pixel
filter3x3[0, 0, 1, 1] = np.random.randn()
# fill the ring of radius 1 around it with a unique value
mask = torch.ones(3, 3, dtype=torch.bool)
mask[1, 1] = 0
filter3x3[0, 0, mask] = np.random.randn()

plt.imshow(filter3x3[0, 0].numpy())
plt.title('Filter')
plt.show()


psi_x = torch.conv2d(x, filter3x3, bias=None, padding=1)
psi_gx = torch.conv2d(gx, filter3x3, bias=None, padding=1)

g_psi_x = rotate(psi_x, r)

plt.imshow(g_psi_x[0, 0].numpy())
plt.title('$g.\\psi(x)$')
plt.show()

plt.imshow(psi_gx[0, 0].numpy())
plt.title('$\\psi(g.x)$')
plt.show()

plt.imshow((g_psi_x-psi_gx)[0, 0].numpy(), vmin=-0.1, vmax=0.1)
plt.title('$g.\\psi(x)-\\psi(g.x)$')
plt.show()

class LiftingConv2d(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):

        super(LiftingConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        # let's create a tensor which stores the learnable filters
        # this layer have `out_channels x in_channels` different learnable filters, each of shape `kernel_size x kernel_size`
        randn_t = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.weight = torch.nn.Parameter(randn_t/np.sqrt(self.out_channels * self.in_channels), requires_grad=True)

        # bias is shared over the 4 rotations
        # it has `out_channels` learnable parameters, one for each independent output
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels), requires_grad=True)

    def build_filter(self):

        # now we build the bigger filter of shape `out_channels x 4 x in_channels x kernel_size x kernel_size` by rotating 4 times
        # it should contain the learnable filter rotated `r` times
        _filter = torch.stack([rotate(self.weight, r) for r in range(4)], dim=1)  # stack concateantes along a new dimension dim

        # now build the `_bias[:, i]` should contain a copy of the learnable bias for each `i=0,1,2,3`
        _bias = torch.stack([self.bias for r in range(4)], dim=1)
        return _filter, _bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, self.kernel_size, self.kernel_size)

        # to be able to use torch.conv2d, we need to reshape the filter and bias to stack together all filters
        _filter = _filter.reshape(self.out_channels * 4, self.in_channels, self.kernel_size, self.kernel_size)
        _bias = _bias.reshape(self.out_channels * 4)

        out = torch.conv2d(x, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=_bias)
        # `out` has now shape `batch_size x out_channels*4 x W x H`
        # we need to reshape it to `batch_size x out_channels x 4 x W x H` to have the shape we expect
        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])

# Let's check if the layer is really equivariant

in_channels = 1
out_channels = 10
kernel_size = 3
batchsize = 1
S = 32

layer = LiftingConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1)
layer.eval()


# generate some input...
xs = torch.linspace(-5, 5, steps=S)
xg, yg = torch.meshgrid(xs, xs, indexing='xy')
x = torch.sin(torch.sqrt(xg * xg + yg * yg))*(yg-xg)/10.

x = x[None, None, ...]
gx = rotate(x, 1)

plt.imshow(x[0, 0].numpy(), cmap='jet')
plt.title('Original Image $x$')
plt.show()

plt.imshow(gx[0, 0].numpy(), cmap='jet')
plt.title('Rotated Image $g.x$')
plt.show()

# compute the output
psi_x = layer(x)
psi_gx = layer(gx)

assert psi_x.shape == (batchsize, out_channels, 4, S, S)

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True, figsize=(16, 4))
for i in range(4):
  axes[i].imshow(psi_x[0, 0, i].detach().numpy(), cmap='jet')
fig.suptitle('$\\psi(x)$')
plt.show()

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True, figsize=(16, 4))
for i in range(4):
  axes[i].imshow(psi_gx[0, 0, i].detach().numpy(), cmap='jet')
fig.suptitle('$\\psi(g.x)$')
plt.show()

# check the model has the right number of parameters
assert layer.weight.numel() == in_channels * out_channels * kernel_size**2
assert layer.bias.numel() == out_channels

class GroupConv2d(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):

        super(GroupConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        # this layer should have `out_channels x in_channels` different learnable filters, each of shape `4 x kernel_size x kernel_size`
        randn_t = torch.randn(self.out_channels, self.in_channels, 4, self.kernel_size, self.kernel_size)
        self.weight = torch.nn.Parameter(randn_t/np.sqrt(self.out_channels * self.in_channels), requires_grad=True)

        # bias is shared over the 4 rotations
        # it has `out_channels` learnable parameters, one for each independent output
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels), requires_grad=True)

    def build_filter(self):

        # now we build the bigger filter of shape `out_channels x 4 x in_channels x 4 x kernel_size x kernel_size` by rotating 4 times
        # it should contain the learnable filter rotated `r` times
        _filter = torch.stack([rotate(self.weight, r).roll(r, dims=-3) for r in range(4)], dim=1)  # stack concateantes along a new dimension dim

        # now build the `_bias[:, i]` should contain a copy of the learnable bias for each `i=0,1,2,3`
        _bias = torch.stack([self.bias for r in range(4)], dim=1)

        return _filter, _bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _filter, _bias = self.build_filter()

        assert _bias.shape == (self.out_channels, 4)
        assert _filter.shape == (self.out_channels, 4, self.in_channels, 4, self.kernel_size, self.kernel_size)

        # to be able to use torch.conv2d, we need to reshape the filter and bias to stack together all filters
        _filter = _filter.reshape(self.out_channels * 4, self.in_channels * 4, self.kernel_size, self.kernel_size)
        _bias = _bias.reshape(self.out_channels * 4)

        # this time, also the input has shape `batch_size x in_channels x 4 x W x H`
        # so we need to reshape it to `batch_size x in_channels*4 x W x H` to be able to use torch.conv2d
        x = x.view(x.shape[0], self.in_channels*4, x.shape[-2], x.shape[-1])

        out = torch.conv2d(x, _filter,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=_bias)

        # `out` has now shape `batch_size x out_channels*4 x W x H`
        # we need to reshape it to `batch_size x out_channels x 4 x W x H` to have the shape we expect
        return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])

class C4CNN(torch.nn.Module):
    def __init__(self, n_classes=10):

        super(C4CNN, self).__init__()
        in_channels = 1
        channels = [8, 16, 32, 64, 128]
        kernel_size = 3

        self.lifting = LiftingConv2d(in_channels=in_channels, out_channels=channels[0], kernel_size = kernel_size, padding=1)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    GroupConv2d(in_channels=channels[nc], out_channels=channels[nc+1], kernel_size = kernel_size, padding=1),
                    torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
                    torch.nn.ReLU()
                ) for nc in range(len(channels[:-1]) - 1)
            ]
        )
        self.output_layer = torch.nn.Sequential(
                    GroupConv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size = kernel_size, padding=1),
                    torch.nn.MaxPool3d(kernel_size=(4,3,3), stride=(1,1,1), padding=(0,0,0)),
                    torch.nn.ReLU())

    def forward(self, x):
        x = self.lifting(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

net = C4CNN()

x = torch.randn(5, 1, 33, 33)
gx = rotate(x, 1)
y = net(x)
gy = net(gx)
print(y.shape)

plt.imshow(y[2, 33, 0].detach().numpy(), cmap='jet')
plt.title('$\\Psi(x)$')
plt.show()
plt.imshow(gy[2, 33, 0].detach().numpy(), cmap='jet')
plt.title('$\\Psi(g.x)$')
plt.show()

import numpy as np
from scipy.special import factorial, sph_harm, assoc_laguerre
from random import randrange, random

def hydrogen_wavefunction(n, l, m, r, theta, phi, a0=1.0):
    """
    Calculates the hydrogen atom wavefunction psi_nlm(r, theta, phi).

    Args:
        n (int): Principal quantum number.
        l (int): Angular momentum quantum number.
        m (int): Magnetic quantum number.
        r (float): Radial coordinate.
        theta (float): Polar angle.
        phi (float): Azimuthal angle.
        a0 (float): Bohr radius (default is 1.0 for simplicity).

    Returns:
        complex: The complex wavefunction value.
    """
    # Radial part R_nl(r)
    rho = r*2/n/a0
    term1 = np.sqrt((2/n/a0)**3*factorial(n-l-1)/factorial(n+l)/n/2)
    term2 = rho**l
    term3 = np.exp(-rho/2)
    term4 = assoc_laguerre(rho, n-l-1, 2*l+1)
    R_nl = term1*term2*term3*term4

    # Angular part Y_lm(theta, phi)
    Y_lm = sph_harm(m, l, phi, theta)

    # Full wavefunction psi_nlm(r, theta, phi)
    psi = R_nl*Y_lm
    return psi

ys = np.linspace(-1., 1., num=201)
yg, zg = np.meshgrid(ys, ys)

r = np.sqrt(yg**2 + zg**2)
theta = np.arctan2(yg, zg)
phi = np.zeros_like(theta)

n = 4
l = 2
m = 0

rand_a = random() * 0.08 + 0.02
rand_k = randrange(4)

psi = hydrogen_wavefunction(n, l, m, r, theta, phi, rand_a)

image = np.real(psi)
image = np.rot90(image, k=rand_k).copy()
image /= np.max(np.abs(image)) + 1e-8

plt.imshow(image, cmap="turbo")
plt.title(rf"$n={n},\,l={l},\,m={m}$")
plt.show()
