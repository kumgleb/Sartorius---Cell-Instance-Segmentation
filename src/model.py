import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
from torchvision.models.resnet import resnet34, resnet50, resnet101


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, 2),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 2),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg["backbone"] == "resnet34":
            self.backbone = resnet34(pretrained=True)
        elif cfg["backbone"] == "resnet50":
            self.backbone = resnet50(pretrained=True)
        elif cfg["backbone"] == "resnet101":
            self.backbone = resnet101(pretrained=True)
        else:
            raise NotImplementedError(
                "Such backbone not implemented. ResNet50 and ResNet101 are available."
            )
        self.backbone.conv1 = nn.Conv2d(
            1,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=1,
            padding=3,
            bias=False,
        )

    def forward(self, x):
        features = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        features.append(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        features.append(x)
        x = self.backbone.layer2(x)
        features.append(x)
        x = self.backbone.layer3(x)
        features.append(x)
        x = self.backbone.layer4(x)
        features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 2)]
            + [Block(128, 64)]
        )
        self.Tconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)
            ]
        )

    def crop(self, x, features):
        _, _, w, h = x.shape
        return transforms.CenterCrop([w, h])(features)

    def forward(self, x, features):
        for i, block in enumerate(self.decoder_blocks):
            x = self.Tconvs[i](x)
            features_for_cat = self.crop(x, features[i])
            x = torch.cat([x, features_for_cat], dim=1)
            x = block(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ResNetEncoder(cfg)
        self.decoder = Decoder(cfg["decoder_channels"])
        self.head = nn.Conv2d(cfg["decoder_channels"][-1], cfg["n_classes"], 1)
        self.retain_dim = cfg["retain_dim"]
        self.out_size = cfg["out_size"]

    def forward(self, x):
        encoded_features = self.encoder(x)
        decoded_features = self.decoder(
            encoded_features[::-1][0], encoded_features[::-1][1:]
        )
        res = self.head(decoded_features)
        if self.retain_dim:
            res = F.interpolate(res, self.out_size)
        return res
