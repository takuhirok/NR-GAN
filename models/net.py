import torch
import torch.nn as nn

from .common import (CustomConv2d, ResidualBlock, OptimizedResidualBlock,
                     global_pooling)


class Generator(nn.Module):
    def __init__(self,
                 latent_dim=128,
                 image_size=32,
                 image_channels=3,
                 channels=128,
                 residual_factor=0.1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.channels = channels
        self.residual_factor = residual_factor

        self.linear1 = nn.Linear(
            latent_dim,
            channels * (image_size // 8) * (image_size // 8))
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='up',
                                    residual_factor=residual_factor)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='up',
                                    residual_factor=residual_factor)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='up',
                                    residual_factor=residual_factor)
        self.relu5 = nn.ReLU()
        self.conv5 = CustomConv2d(channels,
                                  image_channels,
                                  3,
                                  residual_init=False)
        self.act5 = nn.Tanh()

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.latent_dim)

    def forward(self, input):
        output = input
        output = self.linear1(output)
        output = output.view(-1, self.channels, self.image_size // 8,
                             self.image_size // 8)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.relu5(output)
        output = self.conv5(output)
        if hasattr(self, 'act5'):
            output = self.act5(output)
        return output


class Discriminator(nn.Module):
    def __init__(self,
                 image_channels=3,
                 channels=128,
                 residual_factor=0.1,
                 pooling='mean'):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.image_channels = image_channels
        self.residual_factor = residual_factor
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(image_channels,
                                             channels,
                                             3,
                                             residual_factor=residual_factor)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    residual_factor=residual_factor)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    residual_factor=residual_factor)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    residual_factor=residual_factor)
        self.relu5 = nn.ReLU()
        self.linear5 = nn.Linear(channels, 1)

    def forward(self, input):
        output = input
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.relu5(output)
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5(output)
        return out_dis.squeeze()
