import torch
import torch.nn as nn

#what to pay attention to, selection process through channels
class SE_Block(nn.Module): 
    def __init__(self, inputs_g, inputs_x, inputs_inter):
        super(SE_Block, self).__init__()
        self.W_g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Linear(inputs_g , inputs_inter // 16, kernel_size=1, stride=1, padding=0, bias=True), # Fully Connected layer
            nn.ReLU(inplace=True),
            nn.Linear(inputs_g // 16, inputs_inter, kernel_size=1, stride=1, padding=0, bias=True), # Fully Connected layer
            #nn.Sigmoid()
        )

        self.W_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Linear(inputs_x, inputs_inter // 16, kernel_size=1, stride=1, padding=0, bias=True), # Fully Connected layer
            nn.ReLU(inplace=True),
            nn.Linear(inputs_x // 16, inputs_inter, kernel_size=1, stride=1, padding=0, bias=True), # Fully Connected layer
            #nn.Sigmoid()
        )

        self.inter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Linear(inputs_inter , 1 // 16, kernel_size=1, stride=1, padding=0, bias=True), # Fully Connected layer
            nn.ReLU(inplace=True),
            nn.Linear(inputs_inter // 16, 1 , kernel_size=1, stride=1, padding=0, bias=True), # Fully Connected layer
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        inter = self.relu(g1 + x1)
        inter = self.inter(inter)
        return x * inter

# class GSoP_Block(nn.Module): 
#     def __init__(self, F_g, F_l, F_int):
#         super(GSoP_Block, self).__init__()

#     def forward(self, g, x):
        
#         return x * psi

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