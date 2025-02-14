import torch
import torch.nn as nn
import torch.nn.functional as F
import ChannelAttentions 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temp_channels=None, dropout_rate=0.1):
        super().__init__()
        if temp_channels is None:
            temp_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, temp_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(temp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(temp_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dropout_rate=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
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


class BasicAttentionBlock(nn.Module):
    def __init__(self, inputs_g, inputs_x, inputs_inter):
        super(BasicAttentionBlock, self).__init__()
        self.W_g = nn.Sequential( #gating signal with intermediate inputs, higher level features
            nn.Conv2d(inputs_g, inputs_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inputs_inter)
        )

        self.W_x = nn.Sequential( #input channel with intermediate inputs, local feature
            nn.Conv2d(inputs_x, inputs_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inputs_inter)
        )

        self.inter = nn.Sequential(
            nn.Conv2d(inputs_inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        inter = self.relu(g1 + x1)
        inter = self.inter(inter)
        return x * inter


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, 
                 dropout_rate=0.1, att_1="Attention_Gate", att_2="Attention_Gate", att_3="Attention_Gate", att_4="Attention_Gate"):
        super().__init__()
        
        self.use_attention = use_attention
        self.att_1 = att_1
        self.att_2 = att_2
        self.att_3 = att_3
        self.att_4 = att_4

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.inc = ConvBlock(in_channels, 64, dropout_rate=dropout_rate)
        self.down1 = ConvBlock(64, 128, dropout_rate=dropout_rate)
        self.down2 = ConvBlock(128, 256, dropout_rate=dropout_rate)
        self.down3 = ConvBlock(256, 512, dropout_rate=dropout_rate)
        self.down4 = ConvBlock(512, 1024, dropout_rate=dropout_rate)
        
        self.up5 = UpBlock(1024, 512, dropout_rate=dropout_rate)
        self.upconv5 = ConvBlock(1024, 512, dropout_rate=dropout_rate)
        
        self.up4 = UpBlock(512, 256, dropout_rate=dropout_rate)
        self.upconv4 = ConvBlock(512, 256, dropout_rate=dropout_rate)
        
        self.up3 = UpBlock(256, 128, dropout_rate=dropout_rate)
        self.upconv3 = ConvBlock(256, 128, dropout_rate=dropout_rate)
        
        self.up2 = UpBlock(128, 64, dropout_rate=dropout_rate)
        self.upconv2 = ConvBlock(128, 64, dropout_rate=dropout_rate)
        
        self.out = ConvOut(64, out_channels)
        
        if self.att_1 != "None":
            self.att5 = self._get_attention_block(att_1, 512, 512, 256)
        if self.att_2 != "None":
            self.att4 = self._get_attention_block(att_2, 256, 256, 128)
        if self.att_3 != "None":
            self.att3 = self._get_attention_block(att_3, 128, 128, 64)
        if self.att_4 != "None":
            self.att2 = self._get_attention_block(att_4, 64, 64, 32)

    def _get_attention_block(self, att_type, in_channels, inter_channels, out_channels):
        match att_type:
            case "Attention_Gate":
                return BasicAttentionBlock(in_channels, inter_channels, out_channels)
            case "SRM_Attention":
                return ChannelAttentions.SRM_Block(in_channels, inter_channels, out_channels)
            case "GCT_Attention":
                return ChannelAttentions.GCT_Block(in_channels, inter_channels, out_channels)
            case _:
                raise ValueError(f"Unknown attention type: {att_type}")

    def set_grad_cam_target(self, target_layer):
        self.grad_cam_target = target_layer
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
        
        if self.use_attention:
            u5 = self.up5(x5)
            if self.att_1 != "None":
                x4 = self.att5(u5, x4)
            u5 = torch.cat((x4, u5), dim=1)
            u5 = self.upconv5(u5)
            
            u4 = self.up4(u5)
            if self.att_2 != "None":
                x3 = self.att4(u4, x3)
            u4 = torch.cat((x3, u4), dim=1)
            u4 = self.upconv4(u4)
            
            u3 = self.up3(u4)
            if self.att_3 != "None":
                x2 = self.att3(u3, x2)
            u3 = torch.cat((x2, u3), dim=1)
            u3 = self.upconv3(u3)
            
            u2 = self.up2(u3)
            if self.att_4 != "None":
                x1 = self.att2(u2, x1)
            u2 = torch.cat((x1, u2), dim=1)
            u2 = self.upconv2(u2)
        else:
            u5 = self.up5(x5)
            u5 = torch.cat((x4, u5), dim=1)
            u5 = self.upconv5(u5)
            
            u4 = self.up4(u5)
            u4 = torch.cat((x3, u4), dim=1)
            u4 = self.upconv4(u4)
            
            u3 = self.up3(u4)
            u3 = torch.cat((x2, u3), dim=1)
            u3 = self.upconv3(u3)
            
            u2 = self.up2(u3)
            u2 = torch.cat((x1, u2), dim=1)
            u2 = self.upconv2(u2)
        
        u1 = self.out(u2)
        return u1