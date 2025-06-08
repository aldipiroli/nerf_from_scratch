import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = torch.tensor([2**i * torch.pi for i in range(self.num_frequencies)])

    def forward(self, x_in):
        b, N, M, coord_size = x_in.shape
        x = x_in.unsqueeze(-1) * self.frequencies.to(x_in.device)
        sin = torch.sin(x)
        cos = torch.cos(x)
        x_pos_encode = torch.cat([sin, cos], -1)
        x_pos_encode = x_pos_encode.reshape(b, N, M, -1)
        x_pos_encode = torch.cat([x_pos_encode, x_in], -1)
        return x_pos_encode


class NeRFModel(nn.Module):
    def __init__(
        self,
        input_size=6,
        output_size=4,
        embed_size=64,
        num_encode_layers=4,
        num_frequencies_enc=10,
        num_frequencies_dec=4,
    ):
        super(NeRFModel, self).__init__()

        encoder_in_size = 3 + (3 * 2 * num_frequencies_enc)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in_size, embed_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(embed_size, embed_size), nn.ReLU()) for _ in range(num_encode_layers - 1)]
        )

        decoder_in_size = 3 + (3 * 2 * num_frequencies_dec)

        self.decoder = nn.Sequential(
            nn.Linear(embed_size + decoder_in_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, output_size),
        )

        self.positional_encoding_xyz = PositionalEncoding(num_frequencies=num_frequencies_enc)
        self.positional_encoding_d = PositionalEncoding(num_frequencies=num_frequencies_dec)

    def forward(self, data):
        coord = data[..., :3]  # x,y,z coords
        d = data[..., 3:]  # unit direction vector

        coord = self.positional_encoding_xyz(coord)
        d = self.positional_encoding_d(d)

        coord_encoded = self.encoder(coord)
        embeds = torch.cat((coord_encoded, d), -1)

        preds = self.decoder(embeds)
        preds_color = preds[..., :3]
        preds_density = preds[..., 3:4]
        return preds_color, preds_density


if __name__ == "__main__":
    x = torch.rand(1, 10, 20, 6)
    model = NeRFModel(10)
    preds_color, preds_density = model(x)
