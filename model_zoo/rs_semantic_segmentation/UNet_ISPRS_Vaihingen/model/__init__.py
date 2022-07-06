from luojianet_ms import nn
from .unet import UNet


def get_unet(in_channels, n_class):
    return UNet(n_channels=in_channels,n_classes=n_class)

class SegModel(nn.Module):
    def __init__(self,model_network:str,in_channels: int = 3, n_class: int = 6):
        super(SegModel, self).__init__()
        self.in_channels=in_channels
        self.n_class=n_class
        self.model_network=model_network
        self.model=get_unet(in_channels=self.in_channels,n_class=self.n_class)

    def forward(self,x):
        x =self.model(x)
        return x
