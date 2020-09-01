import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as WN


class CLSTM(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_channels):
        super().__init__()
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

        self.rnn = nn.LSTM(n_hidden_channels, n_hidden_channels // 2, 1,
                           batch_first=True, bidirectional=True)
        self.output_projection = nn.Sequential(
                WN(nn.Conv1d(n_hidden_channels, n_out_channels, kernel_size=1)),
                nn.LeakyReLU(0.2, True))

    def forward(self, x):
        x = self.conv_in(x)
        for conv in self.convolutions:
            x = x + conv(x)
        outputs, _ = self.rnn(x.transpose(1, 2))
        outputs = self.output_projection(outputs.transpose(1, 2))
        return outputs


class SpeakerEncoder(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_channels,
                 embedding_size):
        super().__init__()
        self.spk_embeddings = nn.Parameter(torch.FloatTensor(
            embedding_size, n_hidden_channels))
        torch.nn.init.normal_(self.spk_embeddings, mean=0., std=0.5)

        self.encoder = nn.Sequential(
            WN(nn.Conv1d(n_in_channels, n_hidden_channels, 1)),
            nn.LeakyReLU(0.2, True),
            WN(nn.Conv1d(n_hidden_channels, embedding_size, 1)),
            nn.Softmax(dim=1))
        self.projection = nn.Sequential(
            WN(nn.Conv1d(n_hidden_channels, n_out_channels, 1)),
            nn.LeakyReLU(0.2, True))

    def forward(self, x):
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = torch.matmul(x, self.spk_embeddings)
        x = x.transpose(1, 2)

        x = self.projection(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_channels,
                 embedding_size=64):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.ppg_encoder = CLSTM(
                n_in_channels[0], n_out_channels[0], n_hidden_channels[0])
        self.pitch_encoder = CLSTM(
                n_in_channels[1], n_out_channels[1], n_hidden_channels[1])
        self.spk_encoder = SpeakerEncoder(n_in_channels[2], n_out_channels[2],
                                          n_hidden_channels[2], embedding_size)


    def forward(self, x):
        # x: [B, L, C]
        ppg, pitch, spk = torch.split(x, self.n_in_channels, dim=1)
        
        ppg_embedding = self.ppg_encoder(ppg)
        pitch_embedding = self.pitch_encoder(pitch)
        spk_embedding = self.spk_encoder(spk)

        x = torch.cat([ppg_embedding, pitch_embedding, spk_embedding], dim=1)
        return x
