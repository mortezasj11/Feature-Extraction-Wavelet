import numpy as np
#import nibabel as nib
#import skimage.io as io
#import os
import torch.nn.functional as F
import torch
import torch.nn as nn

#from torchsummary import summary
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)


    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)


    
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)
    

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    


    
class AE(nn.Module):
    def __init__(self, args):
        #[32, 38, 45, 54, 64, 76]  #[28, 30, 33, 36, 39, 42, 46] #[24, 24, 24, 24, 24, 24, 24]   # #[32, 16, 16, 16, 16, 16, 24] #self.channels = [32, 35, 38, 41, 45, 49, 53]#1/20 #[28, 30, 33, 36, 39, 42, 46]  1/24.3 #[32, 16, 16, 16, 16, 16, 16]  1/64
        self.model_size = args.model_size
        self.channels = self.model_size*np.array([32, 16, 16, 16, 16, 16, 16])
        if args.mode =='wavelet':
            self.channels = 4*self.channels
        super(AE, self).__init__()
        self.in_channel = args.in_channel
        self.out_channel = args.out_channel
        self.num_classes = args.num_classes

        self.inc = DoubleConv(self.in_channel, self.channels[0])
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4 = Down(self.channels[3], self.channels[4])
        self.down5 = Down(self.channels[4], self.channels[5])
        
        self.up1 = Up(self.channels[5], self.channels[4])
        self.up2 = Up(self.channels[4], self.channels[3])
        self.up3 = Up(self.channels[3], self.channels[2])
        self.up4 = Up(self.channels[2], self.channels[1])
        self.up5 = Up(self.channels[1], self.channels[0])
        
        self.outc = OutConv(self.channels[0], self.out_channel)

        self.cat_class = nn.Sequential(
            nn.Linear(self.model_size*576, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )

    def forward(self, x):
        x = self.inc(x)     
        x = self.down1(x)   
        x = self.down2(x)   
        x = self.down3(x)   
        x = self.down4(x)  
        x_encoded = self.down5(x)  
        #print(x1.shape)
        x = self.up1(x_encoded)    
        x = self.up2(x)     
        x = self.up3(x)     
        x = self.up4(x)     
        x = self.up5(x)     
        x = self.outc(x)
        #print(x_encoded.shape)
        x_encoded_flatten =torch.flatten(x_encoded, start_dim=1)
        label = self.cat_class(x_encoded_flatten)

        return x, label
    
if __name__=='__main__':
    
    model1 = AE(1, 1)
    image = torch.rand(1, 1, 256, 256)
    output, bt = model1(image)
    print()
    print('output.shape: ', output.shape)
    print('bottle_neck.shape: ', bt.shape)
    #summary(model1, (1,512,512))