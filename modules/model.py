import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_classes, upsampling='subpixel'):
        super().__init__()

        self.num_classes = num_classes

        self.upsampling = upsampling

        self.maxpool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()

        self.final_layer = nn.Conv2d(32, self.num_classes, 3, 1, 1)

        self.build_encoder()
        self.build_decoder()

    def build_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            self.activation,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            self.activation,
        )
        return layer

    def build_encoder(self):
        self.E1_layer = self.build_layer(1, 32)
        self.E2_layer = self.build_layer(32, 64)
        self.E3_layer = self.build_layer(64, 128)
        self.E4_layer = self.build_layer(128, 256)

    def build_decoder(self, r=2):
        if self.upsampling == 'subpixel':
            self.up = nn.PixelShuffle(r)

            self.D1a_conv = nn.Conv2d(256, 128*r**2, 3, stride=1, padding=1)
            self.D1b_conv = nn.Conv2d(128+128, 128, 3, stride=1, padding=1)

            self.D2a_conv = nn.Conv2d(128, 64*r**2, 3, stride=1, padding=1)
            self.D2b_conv = nn.Conv2d(64+64, 64, 3, stride=1, padding=1)

            self.D3a_conv = nn.Conv2d(64, 32*r**2, 3, stride=1, padding=1)
            self.D3b_conv = nn.Conv2d(32+32, 32, 3, stride=1, padding=1)

        elif self.upsampling == 'transpose':
            self.up1 = nn.ConvTranspose2d(256, 256, 2, 2, 0)
            self.D1_layer = self.build_layer(256+128, 128)

            self.up2 = nn.ConvTranspose2d(128, 128, 2, 2, 0)
            self.D2_layer = self.build_layer(128+64, 64)

            self.up3 = nn.ConvTranspose2d(64, 64, 2, 2, 0)
            self.D3_layer = self.build_layer(64+32, 32)

    def forward(self, x):
        # Encoder
        E1_out = self.E1_layer(x)

        out = self.maxpool(E1_out)
        E2_out = self.E2_layer(out)

        out = self.maxpool(E2_out)
        E3_out = self.E3_layer(out)

        out = self.maxpool(E3_out)
        bottleneck = self.E4_layer(out)

        # Decoder
        if self.upsampling == 'subpixel':
            out = self.D1a_conv(bottleneck)
            out = self.up(out)
            out = torch.cat([out, E3_out], dim=1)
            D1_out = self.D1b_conv(out)

            out = self.D2a_conv(D1_out)
            out = self.up(out)
            out = torch.cat([out, E2_out], dim=1)
            D2_out = self.D2b_conv(out)

            out = self.D3a_conv(D2_out)
            out = self.up(out)
            out = torch.cat([out, E1_out], dim=1)
            D3_out = self.D3b_conv(out)

        elif self.upsampling == 'transpose':
            out = self.up1(bottleneck)
            out = torch.cat([out, E3_out], dim=1)
            D1_out = self.D1_layer(out)

            out = self.up2(D1_out)
            out = torch.cat([out, E2_out], dim=1)
            D2_out = self.D2_layer(out)

            out = self.up3(D2_out)
            out = torch.cat([out, E1_out], dim=1)
            D3_out = self.D3_layer(out)

        out = self.final_layer(D3_out)

        return out
