
import torch
import pywt


def iWave(imW, wavelet='bior1.3', level=1, mode='zero'):
    out = torch.zeros(  (  imW.shape[0] , 1 , imW.shape[2]*2 , imW.shape[3]*2  )  )
    for i in range(imW.shape[0]):            
        LL, (LH, HL, HH) = imW[i,0,:,:].cpu().detach().numpy(),(imW[i,1,:,:].cpu().detach().numpy(), imW[i,2,:,:].cpu().detach().numpy(), imW[i,3,:,:].cpu().detach().numpy())
        LL, (LH, HL, HH) = pywt.pad(LL,1,'zero'),  (pywt.pad(LH,1,'zero'),pywt.pad(HL,1,'zero'),pywt.pad(HH,1,'zero'))
        im = pywt.idwt2((LL, (LH, HL, HH)), wavelet) 
        out[i,0,:,:] = torch.from_numpy(im)
    return out