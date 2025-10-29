# -*- coding: utf-8 -*-
"""
自编码器模型实现
"""
from huggingface_hub import PyTorchModelHubMixin
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DModel

class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=2):
        super(Encoder, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fl = nn.Flatten()
        self.mlp = nn.Sequential(nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fl(x)
        x = self.mlp(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim=2, out_channels=1):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 128 * 28 * 28))
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, 128, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class AE(nn.Module, PyTorchModelHubMixin):
    """
    Autoencoder model for MNIST dataset(image2D).

    Args:
        - in_channels: int, Number of input channels.
        - embedding_dim: int, Dimension of the embedding space.
        - out_channels: int, Number of output channels.

    Example:

    ```python
    >>> from datasets import load_from_disk
    >>> from utils import ae_encode
    >>> from model import AE
    >>> mnist_ae = AE.from_pretrained("Rosykunai/mnist-ae")

    # Load the dataset
    >>> dataset = load_from_disk("mnist_encoded")
    >>> dataset = dataset['train']
    >>> traindata_raw2d = np.stack(trainset['image2D'])

    # Encode the image
    >>> traindata_ae = ae_encode(mnist_ae, traindata_raw2d)
    ```
    """

    def __init__(self, in_channels=1, embedding_dim=2, out_channels=1):
        super(AE, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, dim=embedding_dim)
        self.decoder = Decoder(dim=embedding_dim, out_channels=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ClassUNet(nn.Module, PyTorchModelHubMixin):
    """
    Class conditional DDPM model for MNIST dataset(image2D).
    """

    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()

        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=28,
            in_channels=1 + class_emb_size,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        B, C, W, H = x.shape
        class_embs = self.class_emb(class_labels).view(B, -1, 1, 1).expand(B, -1, W, H)

        inputs = torch.cat([x, class_embs], dim=1)
        return self.model(inputs, t).sample
