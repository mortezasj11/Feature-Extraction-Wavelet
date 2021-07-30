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
    def __init__(self, in_channel, out_channel, bilinear=True, mode='regular'):
        #[32, 38, 45, 54, 64, 76]  #[24, 24, 24, 24, 24, 24, 24]   # #[32, 16, 16, 16, 16, 16, 24]
        #self.channels = [32, 35, 38, 41, 45, 49, 53]#1/20
        #[28, 30, 33, 36, 39, 42, 46]  1/24.3
        #[32, 16, 16, 16, 16, 16, 16]  1/64
        self.channels = np.array([32, 16, 16, 16, 16, 16, 16])#[28, 30, 33, 36, 39, 42, 46]
        if mode =='wavelet':
            self.channels = 4*self.channels
        super(AE, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channel, self.channels[0])
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4 = Down(self.channels[3], self.channels[4])
        self.down5 = Down(self.channels[4], self.channels[5])
        
        self.up1 = Up(self.channels[5], self.channels[4], bilinear)
        self.up2 = Up(self.channels[4], self.channels[3], bilinear)
        self.up3 = Up(self.channels[3], self.channels[2], bilinear)
        self.up4 = Up(self.channels[2], self.channels[1], bilinear)
        self.up5 = Up(self.channels[1], self.channels[0], bilinear)
        
        self.outc = OutConv(self.channels[0], out_channel)

    def forward(self, x):
        x = self.inc(x)     
        #print(x.shape)
        x = self.down1(x)   
        #print(x.shape)
        x = self.down2(x)   
        #print(x.shape)
        x = self.down3(x)   
        #print(x.shape)
        x = self.down4(x)  
        #print(x.shape)     
        x1 = self.down5(x)  
        #print(x1.shape)
        
        x = self.up1(x1)    
        #print(x.shape)
        x = self.up2(x)     
        #print(x.shape)
        x = self.up3(x)     
        #print(x.shape)
        x = self.up4(x)     
        #print(x.shape)
        x = self.up5(x)     
        #print(x.shape)
        x = self.outc(x)
        #print(x.shape)
        return x, x1
    
if __name__=='__main__':
    
    model1 = AE(1, 1)
    image = torch.rand(1, 1, 256, 256)
    output, bt = model1(image)
    print()
    print('output.shape: ', output.shape)
    print('bottle_neck.shape: ', bt.shape)
    #summary(model1, (1,512,512))