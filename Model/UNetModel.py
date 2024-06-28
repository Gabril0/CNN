import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temp_channels=None):
        super().__init__()
        if temp_channels is None:
            temp_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, temp_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(temp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out,kernel_size=3,stride=1,padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.up(x)

class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inc = ConvBlock(in_channels, 64)
        self.down1 = ConvBlock(64, 128)
        self.down2 = ConvBlock(128, 256)
        self.down3 = ConvBlock(256, 512)
        self.down4 = ConvBlock(512, 1024)

        #no attention implementation
        self.up5 = UpBlock(1024, 512)
        self.upconv5 = ConvBlock(1024, 512)

        self.up4 = UpBlock(512, 256)
        self.upconv4 = ConvBlock(512, 256)

        self.up3 = UpBlock(256, 128)
        self.upconv3 = ConvBlock(256, 128)

        self.up2 = UpBlock(128, 64)
        self.upconv2 = ConvBlock(128, 64)


        #attention implementation
        # self.up5 = UpBlock(1024, 512)
        # self.att5 = AttentionBlock(512, 512, 256)
        # self.upconv5 = ConvBlock(1024,512)

        # self.up4 = UpBlock(512, 256)
        # self.att4 = AttentionBlock(256, 256, 128)
        # self.upconv4 = ConvBlock(512, 256)

        # self.up3 = UpBlock(256, 128)
        # self.att3 = AttentionBlock(128, 128, 64)
        # self.upconv3 = ConvBlock(256,128)

        # self.up2 = UpBlock(128, 64)
        # self.att2 = AttentionBlock(64, 64, 32)
        # self.upconv2 = ConvBlock(128,64)

        self.out = ConvOut(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.maxpool(x1)
        x2 = self.down1(x2)

        x3 = self.maxpool(x2)
        x3 = self.down2(x3)

        x4 = self.maxpool(x3)
        x4 = self.down3(x4)

        x5 = self.maxpool(x4)
        x5 = self.down4(x5)

        #no attention implementation
        u5 = self.up5(x5)
        u5 = torch.concat((x4, u5), dim=1)
        u5 = self.upconv5(u5)

        u4 = self.up4(u5)
        u4 = torch.concat((x3, u4), dim=1)
        u4 = self.upconv4(u4)

        u3 = self.up3(u4)
        u3 = torch.concat((x2, u3), dim=1)
        u3 = self.upconv3(u3)

        u2 = self.up2(u3)
        u2 = torch.concat((x1, u2), dim=1)
        u2 = self.upconv2(u2)

        #attention mechanism implementation
        # u5 = self.up5(x5)
        # x4 = self.att5(u5, x4)
        # u5 = torch.concat((x4, u5), dim=1)
        # u5 = self.upconv5(u5)

        # u4 = self.up4(u5)
        # x3 = self.att4(g=u4, x=x3)
        # u4 = torch.concat((x3, u4), dim=1)
        # u4 = self.upconv4(u4)

        # u3 = self.up3(u4)
        # x2 = self.att3(g=u3, x=x2)
        # u3 = torch.concat((x2, u3), dim=1)
        # u3 = self.upconv3(u3)

        # u2 = self.up2(u3)
        # x1 = self.att2(g=u2, x=x1)
        # u2 = torch.concat((x1, u2), dim=1)
        # u2 = self.upconv2(u2)

        u1 = self.out(u2)
        
        return u1


