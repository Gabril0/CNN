import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temp_channels=None):
        super().__init__()
        if not temp_channels:
            temp_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, temp_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(temp_channels),
            nn.ReLU(inplace=True), #to optimize memory usage
            nn.Conv2d(temp_channels,out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
        
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    def forward(self,x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            ConvBlock(in_channels,out_channels)
        )
    def forward(self,x,y):
        x = self.up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, 
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, y], dim=1)
        return self.up(x)
    
class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self,x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, channels, classes):  
        super().__init__()
        self.channels = channels
        self.classes = classes

        self.inc = (ConvBlock(channels, 64))
        self.down1 = (DownBlock(64, 128))
        self.down2 = (DownBlock(128, 256))
        self.down3 = (DownBlock(256, 512))
        self.down4 = (DownBlock(512, 1024))

        self.up1 = (UpBlock(1024, 512))
        self.up2 = (UpBlock(512, 256))
        self.up3 = (UpBlock(256, 128))
        self.up4 = (UpBlock(128, 64))
        self.out = (ConvOut(64, classes))
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x