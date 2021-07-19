# Imports
import pathlib
import os

import numpy as np
import nibabel as nib
import torch
from skimage.io import imread
from skimage.transform import resize

from inference import predict
from transformations import normalize_01, re_normalize
from unet_recon_ver2 import UNet



# root directory
root = pathlib.Path.cwd() / 'Lung_public' / 'Test'
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
images_names = get_filenames_of_path(root / 'Input')
targets_names = get_filenames_of_path(root / 'Target')


      
# read images and store them in memory
images = [nib.load(img_name) for img_name in images_names]
targets = [nib.load(tar_name) for tar_name in targets_names]


#--newly added because nifti does not have dtype parameter


for i in range(len(images_names)):
    image = nib.load(images_names[i])
    new_data = np.copy(image.get_fdata())
    hd = image.header
    
    new_dtype = np.double
    new_data = new_data.astype(new_dtype)
    image.set_data_dtype(new_dtype)
    images[i]=new_data
    
    
for j in range(len(targets_names)):
    image = nib.load(targets_names[j])
    new_data = np.copy(image.get_fdata())
    hd = image.header
    
    new_dtype = np.double
    new_data = new_data.astype(new_dtype)
    image.set_data_dtype(new_dtype)
    targets[j]=new_data


# Resize images and targets
images_res = [resize(img, (128, 128,1)) for img in images]
resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]


# --load the trained model---

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model
model = UNet(in_channels=1,
             out_channels=1,
             n_blocks=6,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

# load trained model

model_name = '2D_CT_Recon.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name)

model.load_state_dict(model_weights)


# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img



# postprocess function
def postprocess(img: torch.tensor):
    #img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    #img = re_normalize(img)  # scale it to the range [0-255]
    return img


# predict the segmentation maps 
output = [predict(img, model, preprocess, postprocess, device) for img in images_res]

# ----------save resualts in as nii.gz - use ITK to check visual-----

# output directory
out = pathlib.Path.cwd() / 'Prediction'
os.makedirs(out,exist_ok=True)

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
test_images = (out / 'Input')
os.makedirs(test_images)
test_targets =(out / 'Target')
os.makedirs(test_targets)


# prediction results
for i in range (len(output)):
    pred_mask=nib.Nifti1Image(output[i], affine=np.eye(4))
    file_name=(os.path.basename(images_names[i]).split('.', 1)[0])
    filename_path=os.path.join(test_targets,file_name)
    nib.save(pred_mask,filename_path +'.nii.gz')
    
 # test CT images
for i in range (len(images_res)):
    test_image=nib.Nifti1Image(images_res[i], affine=np.eye(4))
    file_name=(os.path.basename(images_names[i]).split('.', 1)[0])
    filename_path=os.path.join(test_images,file_name)
    nib.save(test_image,filename_path +'.nii.gz')
    

