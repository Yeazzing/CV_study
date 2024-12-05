import torch
import torch.nn as nn

class DownConv(nn.Module):
    def __init__(self, input_channel:int, output_channel:int):
        super().__init__()
        
        self.conv = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.act2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):  #resnet
        x = self.conv(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x
    
    
class UpConv(nn.Module):
    def __init__(self, input_channel:int, output_channel:int):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(input_channel, output_channel, 2, 2)
        
    def forward(self, x: torch.Tensor):
        x = self.up(x)
        return x
    
class Unet(nn.Module):
    def __init__(self, input_list: list, output_list: list):
        super().__init__()
        
        self.downconv = nn.ModuleList([DownConv(i, o) for (i, o) in input_list])
        self.upconv = nn.ModuleList([DownConv(i, o) for (i, o) in output_list])
        self.layer_num = len(input_list)
        self.pool = nn.MaxPool2d(2)
        self.midconv = DownConv(input_list[-1][1], input_list[-1][1]*2)
        self.up = nn.ModuleList([UpConv(i, o) for (i, o) in output_list])
        self.final = nn.Conv2d(output_list[-1][1], 3, 1, 1)
        
    def forward(self, x):
        skip = []
        for i in range(self.layer_num):
            x = self.downconv[i](x)
            skip.append(x)
            x = self.pool(x)
            
        x = self.midconv(x)
        
        for i in range(self.layer_num):
            x = self.up[i](x)
            x = torch.concat((x, skip[self.layer_num-i-1]), dim=1)
            x = self.upconv[i](x)
            
        x = self.final(x)
        return x
    
    
if __name__ == "__main__":
    x = torch.randn(4, 3, 512, 512)
    input_list = [
        (3, 64), (64, 128), (128, 256), (256, 512)
    ]
    output_list = [(1024, 512), (512, 256), (256, 128), (128, 64)]
    model = Unet(input_list, output_list)
    
    x = model(x)
    
    print(x.shape)
        