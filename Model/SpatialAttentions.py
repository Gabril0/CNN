#where to pay attention to
import torch
import torch.nn as nn

class RAM_Block(nn.Module):
    def __init__(self, inputs_g, inputs_x, inputs_inter):
        super(RAM_Block, self).__init__()
        self.inputs_g = inputs_g
        self.inputs_x = inputs_x
        self.inputs_inter = inputs_inter
        # Define any additional layers or parameters here

    def forward(self, g, x):
        # Step A: Divide the image into glimpses
        # Implement the logic to divide the image into glimpses
        glimpses = self.divide_into_glimpses(x)

        # Step B: Get the glimpse features
        # Implement the logic to extract features from the glimpses
        glimpse_features = self.extract_glimpse_features(glimpses)

        # Step C: Get the channel features
        # Implement the logic to extract features from the channels
        channel_features = self.extract_channel_features(g)

        # Combine the glimpse features and channel features
        combined_features = torch.cat([glimpse_features, channel_features], dim=1)

        # Apply any additional operations or transformations
        out = self.process_features(combined_features)

        # Multiply the output with x
        out = out * x

        return out

    def divide_into_glimpses(self, x):
        # Implement the logic to divide the image into glimpses
        # Return the glimpses as a tensor
        # For example, you can use a convolutional layer with a specific kernel size and stride to divide the image into smaller patches
        glimpse_size = 4  # Example: divide the image into 4x4 glimpses
        glimpse_channels = self.inputs_g  # Example: use the same number of channels as the input
        glimpse_stride = 2  # Example: use a stride of 2 to reduce the spatial size of the image

        glimpses = x.unfold(2, glimpse_size, glimpse_stride).unfold(3, glimpse_size, glimpse_stride)
        glimpses = glimpses.contiguous().view(x.size(0), self.inputs_g, glimpse_size * glimpse_size, -1)

        return glimpses

    def extract_glimpse_features(self, glimpses):
        # Implement the logic to extract features from the glimpses
        # Return the glimpse features as a tensor
        # For example, you can use a convolutional layer to extract features from each glimpse
        glimpse_features = self.glimpse_conv(glimpse_features)
        glimpse_features = glimpse_features.view(glimpse_features.size(0), -1)

        return glimpse_features

    def extract_channel_features(self, g):
        # Implement the logic to extract features from the channels
        # Return the channel features as a tensor
        # For example, you can use a convolutional layer to extract features from each channel
        channel_features = self.channel_conv(g)
        channel_features = channel_features.view(channel_features.size(0), -1)

        return channel_features

    def process_features(self, combined_features):
        # Implement any additional operations or transformations on the combined features
        # Return the processed features as a tensor
        # For example, you can use a linear layer to combine the glimpse features and channel features
        out = self.combine_features(combined_features)
        out = self.relu(out)

        return out

    def glimpse_conv(self, x):
        # Implement a convolutional layer to extract features from the glimpses
        glimpse_conv = nn.Conv2d(self.inputs_g, self.inputs_inter, kernel_size=3, stride=1, padding=1, bias=True)
        out = glimpse_conv(x)
        out = nn.functional.relu(out)

        return out

    def channel_conv(self, x):
        # Implement a convolutional layer to extract features from the channels
        channel_conv = nn.Conv2d(self.inputs_g, self.inputs_inter, kernel_size=3, stride=1, padding=1, bias=True)
        out = glimpse_conv(x)
        return out

    
class GENet_Block(nn.Module):
    def __init__(self, inputs_g, inputs_x, inputs_inter):


    def forward(self, g, x):

        return out * x
    
class ViT_Block(nn.Module):
    def __init__(self, inputs_g, inputs_x, inputs_inter):


    def forward(self, g, x):

        return out * x
    

