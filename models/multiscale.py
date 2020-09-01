import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, downsample_scales=[4, 4, 4, 4], max_channels=512):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList()
        self.discriminator += [nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15)),
            nn.LeakyReLU(0.2, True),
        )]
        
        # add downsample layers
        in_channels = 16
        for scale in downsample_scales:
            out_channels = min(in_channels * scale, max_channels)
            self.discriminator += [torch.nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=scale * 10 + 1,
                    stride=scale,
                    padding=scale * 5,
                    groups=in_channels // 4)),
                nn.LeakyReLU(0.2, True)
            )]
            in_channels = out_channels
        
        self.discriminator += [
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels,
                                     kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, True)
            ),
            nn.utils.weight_norm(nn.Conv1d(out_channels, 1, kernel_size=3,
                                           stride=1, padding=1))
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[: -1]


class MultiScaleDiscriminator(nn.Module):
    def __init__(self,
                 multiscales=[2, 2, 2],
                 downsample_scales=[4, 4, 4, 4],
                 max_channels=512):
        super(MultiScaleDiscriminator, self).__init__()

        self.downsamples = nn.ModuleList([nn.AvgPool1d(
            scale * 2, stride=scale, padding=scale // 2, count_include_pad=False)
            for scale in multiscales])
        self.discriminators = nn.ModuleList([
             Discriminator(downsample_scales, max_channels)
             for _ in range(len(multiscales))
        ])

    def forward(self, x):
        scores, feats = list(), list()

        inputs = [x]
        for downsample in self.downsamples[: -1]:
            x = downsample(x)
            inputs.append(x)

        for x, layer in zip(inputs, self.discriminators):
            score, feat = layer(x)
            scores.append(score)
            feats.append(feat)
            
        return scores, feats
