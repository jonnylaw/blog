# Variational auto encoder

# %% Load the mnist dataset
import torch
import torchvision
import torchvision.transforms as transforms

mnist = torchvision.datasets.MNIST(root='../data', download=True)
# %%
type(mnist)

# %%
