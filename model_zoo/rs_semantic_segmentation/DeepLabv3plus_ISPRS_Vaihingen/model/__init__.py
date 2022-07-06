from luojianet_ms import nn
from .deeplabv3p import DeepLabV3Plus

def get_deeplabv3p(in_channels, n_class):
    return DeepLabV3Plus(num_classes=n_class, output_stride=16,aspp_atrous_rates=[1, 12, 24, 36])

class SegModel(nn.Module):
    def __init__(self,model_network:str,in_channels: int = 3, n_class: int = 6):
        super(SegModel, self).__init__()
        self.in_channels=in_channels
        self.n_class=n_class
        self.model_network=model_network
        self.model=get_deeplabv3p(in_channels, n_class)
    def forward(self,x):
        x =self.model(x)
        return x
