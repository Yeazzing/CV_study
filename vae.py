import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.batn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.batn2 = nn.BatchNorm2d(output_channel)
        self.act2 = nn.ReLU()
        
        self.dowm = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        
    def forward(self, x):
        r = self.dowm(x)
        x = self.conv1(x)
        x = self.batn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batn2(x)
        x = self.act2(x)
        x += r
        return x

class UpSample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.batn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.batn2 = nn.BatchNorm2d(output_channel)
        self.act2 = nn.ReLU()
        
        self.dowm = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        
    def forward(self, x):
        r = self.dowm(x)
        x = self.conv1(x)
        x = self.batn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batn2(x)
        x = self.act2(x)
        x += r
        return x

class UpConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channel, output_channel, 2, 2)
        
    def forward(self, x):
        x = self.up(x)
        return x


class Vae(nn.Module):
    def __init__(self, input_list, output_list):
        super().__init__()
        self.input_list = input_list
        self.output_list = output_list
        self.num_layer = len(input_list)
        self.downconv = nn.ModuleList([DownSample(i,o) for (i,o) in self.input_list])
        self.upconv = nn.ModuleList([UpSample(i,o) for (i,o) in self.output_list])
        self.up = nn.ModuleList([UpConv(i,o) for (i,o) in self.output_list])
        self.midconv = DownSample(input_list[self.num_layer-1][1], 2*input_list[self.num_layer-1][1])  #optional
        self.maxpool = nn.MaxPool2d(2)
        self.finalconv = nn.Conv2d(output_list[self.num_layer-1][1], 3, 3, 1, 1)
    
        #self.mu_conv = nn.Conv2d(2*input_list[self.num_layer-1][1], input_list[self.num_layer-1][1], 3, 1, 1)
        #self.var_conv = nn.Conv2d(2*input_list[self.num_layer-1][1], input_list[self.num_layer-1][1], 3, 1, 1)
    
    def sample(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        var = torch.exp(logvar)
        std = torch.exp(0.5 * logvar)
        #mu = self.mu_conv(x)
        #std = self.var_conv(x)
        
        #z = mu + std * torch.randn(std.shape).to(std.device)

        z = mean + std * torch.randn(std.shape).to(std.device)
        
        return z, mu, std, var, logvar
        
    def forward(self, x):
        for i in range(self.num_layer):
            x = self.downconv[i](x)
            x = self.maxpool(x)
            
        x = self.midconv(x)
        
        z, mu, std, var, logvar = self.sample(x)
        
        for j in range(self.num_layer):
            z = self.up[j](z)
            z = self.upconv[j](z)
        
        recon = self.finalconv(z)
        return recon, z, mu, std, var, logvar

    @torch.no_grad
    def inference(self, x):
        for i in range(self.num_layer):
            x = self.downconv[i](x)
            x = self.maxpool(x)
            
        x = self.midconv(x)
        
        z, mu, std, var, logvar = self.sample(x)
        
        for j in range(self.num_layer):
            z = self.up[j](z)
            z = self.upconv[j](z)
        
        recon = self.finalconv(z)
        return recon

    
    
if __name__== '__main__':
    x =torch.randn(4, 3, 256, 256)
    input_list = [(3, 64), (64, 128), (128, 256), (256, 512)]
    output_list = [(512, 512), (512, 256), (256, 128), (128, 64)]
    
    model = Vae(input_list, output_list)
    output, z, mu, std = model(x)
    
    sample = model.inference(x)
    
    print(output.shape)