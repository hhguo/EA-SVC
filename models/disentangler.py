import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as WN


class Disentangler(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_channels=256):
        super(Disentangler, self).__init__()
        self.conv_in = WN(nn.Conv1d(n_in_channels, n_hidden_channels, kernel_size=1))

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                WN(nn.Conv1d(n_hidden_channels, n_hidden_channels,
                             kernel_size=3, padding=1)),
                nn.LeakyReLU(0.2, True),
                WN(nn.Conv1d(n_hidden_channels, n_hidden_channels,
                             kernel_size=3, padding=1))
                )
            convolutions.append(conv_layer)
        convolutions.append(nn.LeakyReLU(0.2, True))
        self.convolutions = nn.ModuleList(convolutions)

        self.conv_out = WN(nn.Conv1d(n_hidden_channels, n_out_channels, kernel_size=1))

    def forward(self, x):
        x = self.conv_in(x)
        for conv in self.convolutions:
            x = x + conv(x)
        outputs = self.conv_out(x)
        return outputs
