import torch
import torchvision.transforms.functional
from torch import nn


class DownSample(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return self.act2(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.up  = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, encoder_list: list, decoder_list: list):
        super().__init__()
        
        self.encoder_list = encoder_list
        self.decoder_list = decoder_list
        
        self.channel_len = len(encoder_list)
        
        
        self.down_conv = nn.ModuleList([DownSample(i, o) for i, o in
                                        encoder_list])
        
        self.pool = nn.MaxPool2d(2)
        
        self.up_conv = nn.ModuleList([DownSample(i, o) for i, o in
                                        decoder_list])
        
        self.middle_conv = DownSample(self.encoder_list[-1][1], int(self.encoder_list[-1][1]*2))
        
        self.up = nn.ModuleList([UpSample(i, o)for i , o in 
                                    decoder_list])
        
        self.final_conv =  nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        skip_connections = []
        
        for i in range(self.channel_len):
            x = self.down_conv[i](x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.middle_conv(x)
            
        for i in range(self.channel_len):
            x = self.up[i](x)
            x = torch.cat([x, skip_connections[self.channel_len-1-i]], dim=1)
            x = self.up_conv[i](x)
        
        x = self.final_conv(x)
            
        return x
    
    
if __name__ == '__main__':
    sample = torch.randn(4,3,256, 256) # B,C,H,W
    
    encoder_channels = [(3, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
    decoder_channels = [(2048, 1024), (1024, 512),(512, 256), (256, 128), (128, 64)]
    
    model = UNet(encoder_channels, decoder_channels)

    
    
    output = model(sample)
    print(output.shape)