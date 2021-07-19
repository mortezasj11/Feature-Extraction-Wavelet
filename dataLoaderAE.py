from os.path import splitext
from os import listdir
from os.path import join
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
import pywt


class Dataset_CTLungSeg(Dataset):
    def __init__(self, CT_dir, Lung_dir, isTrain = True, mode='regular', OnlyLung = False): 
        self.OnlyLung = OnlyLung
        self.mode = mode.lower()
        self.isTrain = isTrain
        self.CT_dir = CT_dir
        self.Lung_dir = Lung_dir
        self.ids = [file for file in listdir(CT_dir) if not file.startswith('.')]  

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocessCT(cls, im, minn=-600.0, maxx=200.0):
        im = np.array(im)   #(5,512,512)
        im = np.clip(im, minn , maxx)
        im = (im - minn)/(maxx-minn)      
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
        return im

    @classmethod
    def preprocessLabel(cls, im):
        im = np.array(im)   #(5,512,512)    
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
        return im

    @classmethod
    def Wave(cls, im, wavelet='bior1.3', level=1, mode='zero'):
        LL, (LH, HL, HH) = pywt.dwt2(im, wavelet, mode )
        LL, (LH, HL, HH) = LL[1:-1,1:-1], (LH[1:-1,1:-1], HL[1:-1,1:-1], HH[1:-1,1:-1]) 
        return LL, (LH, HL, HH)

    @classmethod
    def iWave(cls, coeff, wavelet='bior1.3', level=1, mode='zero'):
        LL, (LH, HL, HH) = coeff
        LL, (LH, HL, HH) = pywt.pad(LL,1,'zero'),  (pywt.pad(LH,1,'zero'),pywt.pad(HL,1,'zero'),pywt.pad(HH,1,'zero'))
        im = pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3') 
        return im




    def transform(self, CT, Label):
        
        # Horizontal and vertical flip
        if torch.rand(1) < 0.5:
            CT = TF.hflip(CT)
            Label = TF.hflip(Label)

        # scaling
        if torch.rand(1) < 1.0:
            #affine_params = tt.RandomAffine(0).get_params((0, 0), (0, 0), (0.85, 1.15), (0, 0), img_size=(512,512))
            affine_params = tt.RandomAffine(0).get_params((-10, 10), (0.05, 0.05), (0.85, 1.15), (-5, 5),img_size=(512,512))
            CT = TF.affine(CT, *affine_params)
            Label = TF.affine(Label, *affine_params)

        '''
        if torch.rand(1) < 0.5:
            CT = TF.vflip(CT)
            Label = TF.vflip(Label)
        '''
        # Rotation
        if torch.rand(1) < 0.5:
            randi = torch.randint(0,360,(1,)).item()
            CT = TF.rotate(CT, randi)
            Label = TF.rotate(Label, randi)
        
        # Cropping
        #i, j, h, w = transforms.RandomCrop.get_params(CT, output_size=(420, 420)) #(0, 0, 512, 512)
        #CT  = TF.crop(CT , i , j , h , w)
        #Label = TF.crop(Label, i , j , h , w)

        #brightness
        if torch.rand(1) < 0.5:
            randi = 0.01*(2*torch.rand(1)-1)   # give uniform random between (-0.05, +0.05)
            CT = TF.adjust_brightness(CT, 1 + randi )

        return CT, Label


    def __getitem__(self, i):
        idx = self.ids[i]
        Lung_file = join( self.Lung_dir , idx )
        CT_file = join(self.CT_dir , idx )

        # Loading        
        Lung = np.load(Lung_file)
        CT = np.load(CT_file)

        # Normalizing
        CT = self.preprocessCT(CT)
        Lung = self.preprocessLabel(Lung)


        # Data augmentation
        if self.isTrain:
            CT, Lung = self.transform(  torch.from_numpy(CT), torch.from_numpy(Lung)  )
            if self.mode =='wavelet' and self.OnlyLung ==False:               
                CT, Lung = self.Wave( CT, wavelet='bior1.3', level=1, mode='zero'), self.Wave( Lung, wavelet='bior1.3', level=1, mode='zero')
            if self.mode =='wavelet' and self.OnlyLung ==True: 
                CT, Lung = self.Wave( CT*Lung, wavelet='bior1.3', level=1, mode='zero'), self.Wave( CT*Lung, wavelet='bior1.3', level=1, mode='zero')
            CT, Lung = CT.type(torch.FloatTensor), Lung.type(torch.FloatTensor)
        else:
            CT, Lung = torch.from_numpy(CT), torch.from_numpy(Lung)
            if self.mode =='wavelet' and self.OnlyLung ==False:
                CT, Lung = self.Wave( CT, wavelet='bior1.3', level=1, mode='zero'), self.Wave( Lung, wavelet='bior1.3', level=1, mode='zero')  
            if self.mode =='wavelet' and self.OnlyLung ==True:     
                CT, Lung = self.Wave( CT*Lung, wavelet='bior1.3', level=1, mode='zero'), self.Wave( CT*Lung, wavelet='bior1.3', level=1, mode='zero')    
            CT, Lung = CT.type(torch.FloatTensor), Lung.type(torch.FloatTensor)
        if self.mode !='wavelet' and self.OnlyLung ==True:
            CT = torch.matmul(CT, Lung)

        return CT, CT

