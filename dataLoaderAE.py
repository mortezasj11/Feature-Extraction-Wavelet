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
import pandas as pd
import random


class Random_Slice_Selection():
    def __init__(self ,slices, random_number = 10):
        self.slices = slices
        self.random_number = random_number
        
    def give_sorted_nonrepetitive_random_int(self, start, end, number):
        # give_sorted_nonrepetitive_random_int
        a = np.random.rand(start,end,number)
        a = np.array(list(range(end-start)))
        a += start
        random.shuffle(a )
        return np.sort(a[:number])
    
    def __call__(self):
        s1, e1, s2, e2 = self.slices
        d1, d2 = e1-s1, e2-s2
        if d2 == 0:
            return self.give_sorted_nonrepetitive_random_int(s1, e1, self.random_number)
        else:
            random_number_1 = int( (d1/(d2+d1))*self.random_number )
            random_number_2 = self.random_number - random_number_1
            print(random_number_1,random_number_2)
            l1 = self.give_sorted_nonrepetitive_random_int(s1,e1,random_number_1)
            l2 = self.give_sorted_nonrepetitive_random_int(s2,e2,random_number_2)
            return np.sort(np.append(l1,l2))




class Dataset_CTLungSeg(Dataset):
    def __init__(self, args ): 
        self.OnlyLung = args.OnlyLung
        self.mode = args.mode.lower()
        self.category = args.category
        self.df = args.df
        self.dfTumorSlide = args.dfTumorSlide
        self.in_channel = args.in_channel

        self.isTrain = args.isTrain
        if self.isTrain:
            self.CT_dir = args.CT_dir_train
            self.Lung_dir = args.Seg_dir_train
        else:
            self.CT_dir = args.CT_dir_val
            self.Lung_dir = args.Seg_dir_val
            
        self.ids = [file for file in listdir(self.CT_dir) if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocessCT(cls, im, minn=-800.0, maxx=150.0):
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
        LL, (LH, HL, HH) = pywt.dwt2(im[0,:,:], wavelet, mode )
        LL, (LH, HL, HH) = LL[1:-1,1:-1], (LH[1:-1,1:-1], HL[1:-1,1:-1], HH[1:-1,1:-1]) 
        out = torch.zeros((4,LL.shape[0], LL.shape[1]))
        out[0,:,:] = torch.from_numpy(LL)
        out[1,:,:] = torch.from_numpy(LH)
        out[2,:,:] = torch.from_numpy(HL)
        out[3,:,:] = torch.from_numpy(HH)
        return out

    @classmethod
    def iWave(cls, imW, wavelet='bior1.3', level=1, mode='zero'):
        out = torch.zeros(  (  imW.shape[0] , 1 , imW.shape[2]*2 , imW.shape[3]*2  )  )
        for i in range(imW.shape[0]):            
            LL, (LH, HL, HH) = imW[i,0,:,:],(imW[i,1,:,:], imW[i,2,:,:], imW[i,3,:,:])
            LL, (LH, HL, HH) = pywt.pad(LL,1,'zero'),  (pywt.pad(LH,1,'zero'),pywt.pad(HL,1,'zero'),pywt.pad(HH,1,'zero'))
            im = pywt.idwt2((LL, (LH, HL, HH)), wavelet) 
            out[i,0,:,:] = im
        return out




    def transform(self, CT, Seg, resize = 192):
        
        # Horizontal and vertical flip
        #if torch.rand(1) < 0.5:
        #    CT = TF.hflip(CT)
        #    Seg = TF.hflip(Seg)

        # scaling
        if self.isTrain:
            affine_params = tt.RandomAffine(0).get_params((-45, 45), (0.07, 0.07), (0.80, 1.20), (-10, 10),img_size=(512,512))
            CT = TF.affine(CT, *affine_params)
            Seg = TF.affine(Seg, *affine_params)


        resize_transformm = tt.Resize((resize, resize))
        CT = resize_transformm(CT)
        Seg = resize_transformm(Seg)

        # Rotation
        #if torch.rand(1) < 0.2:
        #    randi = torch.randint(0,360,(1,)).item()
        #    CT = TF.rotate(CT, randi)
        #    Seg = TF.rotate(Seg, randi)


        #brightness
        #if torch.rand(1) < 0.5:
        #    randi = 0.01*(2*torch.rand(1)-1)   # give uniform random between (-0.05, +0.05)
        #    CT = TF.adjust_brightness(CT, 1 + randi )

        return CT, Seg


    def LabelPatient(self, patient, category):
        label = str(  self.df[self.df['PatientID']==patient][category].item()  )
        
        if category == 'Clinical.Stage.run':      
            if label == 'IA':
                return 0    
            elif label == 'IB':
                return 1        
            elif label in ['IIA', 'IIB'] :
                return 2       
            elif label in ['IIIA', 'IIIB', 'IV'] :
                return 3
            
        if category == 'Pathological.T.Stage':
            if label == '1a':
                return 0   
            elif label == '1b':
                return 1        
            elif label in ['2a', '2b'] :
                return 2        
            elif label in ['3', '4'] :
                return 3

    def GetSliceNumber(self, df, file_name, split=','):
        slicelist = df[df['File Name']==file_name]['Tumor Slides'].item()
        slicelist = str(slicelist).split(split)
        start1 = int(slicelist[0].split('(')[-1])
        end1 = int(slicelist[1].split(')')[0])
        start2, end2 = 0, 0                 
        if len(slicelist)==4:
            start2 = int(slicelist[2].split('(')[-1])
            end2 = int(slicelist[3].split(')')[0])   
        return start1,end1,start2,end2


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
        CT, Lung = self.transform(  torch.from_numpy(CT), torch.from_numpy(Lung) )
        '''
        if self.mode =='wavelet' and self.OnlyLung ==False:               
            CT, Lung = self.Wave( CT, wavelet='bior1.3', level=1, mode='zero'), self.Wave( Lung, wavelet='bior1.3', level=1, mode='zero')
        if self.mode =='wavelet' and self.OnlyLung ==True: 
            CT, Lung = self.Wave( CT*Lung, wavelet='bior1.3', level=1, mode='zero'), self.Wave( CT*Lung, wavelet='bior1.3', level=1, mode='zero')
        CT, Lung = CT.type(torch.FloatTensor), Lung.type(torch.FloatTensor)

        if self.mode !='wavelet' and self.OnlyLung ==True:
            CT = torch.matmul(CT, Lung)
        '''


        patient_name = idx[:-4]
        slices = self.GetSliceNumber(self.dfTumorSlide, patient_name)

        max_slide_number = self.in_channel
        slice_numbers = Random_Slice_Selection(slices, max_slide_number)

        
        sorted_slices = slice_numbers()
        CT = CT[sorted_slices,:,:]
        Lung = Lung[sorted_slices,:,:]

        CT_   = torch.zeros(max_slide_number, 192,192)
        Lung_ = torch.zeros(max_slide_number, 192,192)
        CT_[:len(sorted_slices),:,:]= CT
        Lung_[:len(sorted_slices),:,:] = Lung

        label = self.LabelPatient('Sandy1_'+ patient_name, self.category)
        label = torch.tensor(label)    

        return CT_, label
