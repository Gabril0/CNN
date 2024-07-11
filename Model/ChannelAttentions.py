import torch
import torch.nn as nn
import torch.functional as F

#what to pay attention to, selection process through channels
class SE_Block(nn.Module):
    def __init__(self, input_g, input_x, intermediate_channels):
        super(SE_Block, self).__init__()
        self.avg_pool_g = nn.AdaptiveAvgPool2d(1)
        self.fc_g = nn.Sequential(
            nn.Conv2d(input_g, intermediate_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels // 16, intermediate_channels, kernel_size=1)
        )

        self.avg_pool_x = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Sequential(
            nn.Conv2d(input_x, intermediate_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels // 16, intermediate_channels, kernel_size=1)
        )

        self.fc_inter = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels // 16, input_x, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.avg_pool_g(g)
        g1 = self.fc_g(g1)

        x1 = self.avg_pool_x(x)
        x1 = self.fc_x(x1)

        inter = g1 + x1
        inter = self.fc_inter(inter)

        return x * inter

class GSoP_Block(nn.Module):
    def __init__(self, input_g, input_x, intermediate_channels):
        super(GSoP_Block, self).__init__()
        
        self.conv_g1 = nn.Conv2d(input_g, intermediate_channels // 16, kernel_size=1)
        self.cov_pool_g = CovariancePooling(intermediate_channels // 16)
        self.row_conv_g = RowWiseConvolution(intermediate_channels // 16)
        self.conv_g2 = nn.Conv2d(intermediate_channels // 16, intermediate_channels, kernel_size=1)

        self.conv_x1 = nn.Conv2d(input_x, intermediate_channels // 16, kernel_size=1)
        self.cov_pool_x = CovariancePooling(intermediate_channels // 16)
        self.row_conv_x = RowWiseConvolution(intermediate_channels // 16)
        self.conv_x2 = nn.Conv2d(intermediate_channels // 16, intermediate_channels, kernel_size=1)

        self.fc_inter = nn.Sequential(
            nn.Conv2d(intermediate_channels, intermediate_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels // 16, input_x, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.conv_g1(g)  # Conv2d
        g1 = self.cov_pool_g(g1)  # Covariance pooling
        g1 = self.row_conv_g(g1)  # Row-wise convolution
        g1 = self.conv_g2(g1)  # Conv2d

        x1 = self.conv_x1(x)  # Conv2d
        x1 = self.cov_pool_x(x1)  # Covariance pooling
        x1 = self.row_conv_x(x1)  # Row-wise convolution
        x1 = self.conv_x2(x1)  # Conv2d

        inter = g1 + x1
        inter = self.fc_inter(inter) 
        return x * inter  


# class SRM_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(SRM_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi
    
# class GCT_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(GCT_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi
    
# class ECA_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(ECA_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi
    
# class FCA_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(FCA_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi
    
# class ENC_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(ENC_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi