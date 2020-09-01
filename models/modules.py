import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=3, dilation=3**i, padding=3**i)),
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=3, dilation=1, padding=1)),
            )
            for i in range(4)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor):

        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(input_size, output_size, upsample_factor * 2,
                                   upsample_factor, padding=upsample_factor // 2)
        self.layer = weight_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs
