import torch.nn as nn
from monai.networks.nets import UNet

class Custom3DUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,  # model operates in 3D space
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=args.features,  # (32, 64, 128)
            strides=args.strides,  # (2, 2)
            num_res_units=2, # uses 2 residual units per layer
            norm=args.normalization,  # 'batch'
        )
    
    def forward(self, x):
        # x = self.model(x)
        # return x.argmax(dim=1)  # Apply argmax along the channel dimension to get the label map
        return self.model(x)

