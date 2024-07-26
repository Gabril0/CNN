import torch
import torch.nn as nn
import torch.functional as F

#what to pay attention to, selection process through channels
class SRM_Block(nn.Module):
    def __init__(self, inputs_g, inputs_x, inputs_inter):
        super(SRM_Block, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.W_g = nn.Sequential(
            nn.Conv2d(inputs_g * 2, inputs_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ConvTranspose2d(inputs_inter, inputs_inter, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(inputs_inter)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(inputs_x * 2, inputs_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ConvTranspose2d(inputs_inter, inputs_inter, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(inputs_inter)
        )

        self.inter = nn.Sequential(
            nn.Conv2d(inputs_inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        avg_pooled_g = self.avg_pool(g)
        avg_pooled_x = self.avg_pool(x)
        std_g = torch.std(g, dim=[2, 3], keepdim=True)
        std_x = torch.std(x, dim=[2, 3], keepdim=True)
        
        combined_g = torch.cat((avg_pooled_g, std_g), dim=1)
        combined_x = torch.cat((avg_pooled_x, std_x), dim=1)
        
        out_g = self.W_g(combined_g)
        out_x = self.W_x(combined_x)
        
        combined = out_g + out_x
        out = self.inter(combined)
        
        if out.size() != x.size():
            out = nn.functional.interpolate(out, size=x.size()[2:])
        
        return out * x


    
# class GCT_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(GCT_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi




