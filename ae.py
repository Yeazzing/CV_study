import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.act2 = nn.ReLU()
        self.resconv = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        
    def forward(self, x):
        r = self.resconv(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x += r
        x = self.act2(x)
        return x
    
class UpSample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layer(x)
        
class Up(nn.Module):
    def __init__(self, input_channel, output_channel ):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channel, output_channel, 2, 2)
        
    def forward(self, x):
        x = self.up(x)
        return x

class AE(nn.Module):
    def __init__(self, input_list, output_list ):
        super().__init__()
        self.input_list = input_list
        self.output_list = output_list
        self.num_layer = len(self.input_list)
        self.downconv = nn.ModuleList([DownSample(i,o) for (i,o) in self.input_list])
        self.maxpool = nn.MaxPool2d(2)
        self.midconv = DownSample(self.input_list[self.num_layer-1][1], self.input_list[self.num_layer-1][1]*2)
        self.upconv = nn.ModuleList([UpSample(i,o) for (i,o) in self.output_list])
        self.up = nn.ModuleList([Up(i,o) for (i,o) in self.output_list])
        self.finalconv = nn.Conv2d(output_list[self.num_layer-1][1], 3, 1)
        
    def forward(self, x):
        for i in range(self.num_layer):
            x = self.downconv[i](x)
            x = self.maxpool(x)
            
        x = self.midconv(x)
        
        for j in range(self.num_layer):
            x = self.up[j](x)
            x = self.upconv[j](x)
        
        x = self.finalconv(x)
        
        return x
            
            
            
if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256)
    input_list = [(3, 64), (64, 128), (128, 256), (256, 512)]
    output_list = [(1024, 512), (512, 256), (256, 128), (128, 64)]
    
    model = AE(input_list, output_list)
    
    x = model(x)
    print(x.shape)