import torch
import torch.nn as nn


class NeRFModel(nn.Module):
    def __init__(self, input_size=6, output_size=4, embed_size=256, num_encode_layers=8):
        super(NeRFModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, embed_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(embed_size, embed_size), nn.ReLU()) for _ in range(num_encode_layers - 1)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_size + 3, embed_size // 2), nn.ReLU(), nn.Linear(embed_size // 2, output_size)
        )

    def forward(self, data):
        coord = data[..., :3]  # x,y,z coords
        d = data[..., 3:]  # unit direction vector

        coord_encoded = self.encoder(coord)
        embeds = torch.cat((coord_encoded, d), -1)

        preds = self.decoder(embeds)
        preds_color = preds[..., :3]
        preds_density = preds[..., 3:4]
        return preds_color, preds_density


if __name__ == "__main__":
    x = torch.rand(1, 10, 20, 6)
    model = NeRFModel()
    preds_color, preds_density = model(x)
