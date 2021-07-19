# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:35:53 2021

@author: MBSaad
"""
import pathlib

import torch
from AE import AE
from dataLoaderAE import Dataset_CTLungSeg
from trainer import Trainer
from trainer_tensorboard import Trainer_Tensorboard
from torch.utils.tensorboard import SummaryWriter ### RUN pip install tensorboard==2.4.0
from torch.utils.data import DataLoader, random_split


mode = 'regular' # 'wavelet'





# train_dataset
dataset_train = Dataset_CTLungSeg('/Data/MoriRichardProject/TrainValTest/CT_Tr', '/Data/MoriRichardProject/TrainValTest/Label_Tr', isTrain=True, mode=mode)
n_train = len(dataset_train)
dataset_train, _ = random_split(dataset_train, [n_train, 0]) # maybe is not needed
train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
# val_dataset
dataset_val = Dataset_CTLungSeg('/Data/MoriRichardProject/TrainValTest/CT_Va', '/Data/MoriRichardProject/TrainValTest/Label_Va', isTrain=False, mode=mode)
n_val = len(dataset_val)
print('n_train, n_val', n_train, n_val)
dataset_val, _ = random_split(dataset_val, [n_val, 0]) # maybe is not needed
val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


# device
if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')
   
#device = torch.device('cpu')
# model

model = AE(in_channel = 1, out_channel =1).to(device) if mode=='regular' else AE(in_channel = 4, out_channel =4).to(device)

# criterion
criterion = torch.nn.MSELoss()        #criterion = torch.nn.CrossEntropyLoss()               #criterion = DiceLoss(predict,target)                #criterion = torch.nn.L1Loss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# trainer

trainer = Trainer_Tensorboard(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_loader,
                  validation_DataLoader=val_loader,
                  lr_scheduler=None,
                  epochs=50,
                  epoch=0,
                  notebook=False)

# start training
training_losses, validation_losses, learning_rates = trainer.run_trainer()


# plot training
'''
fig = plot_training(training_losses,
                    validation_losses,
                    learning_rates,
                    gaussian=True,
                    sigma=1,
                    figsize=(10,4))
# save fig
fig.savefig('2D_CT_Recon.png')
'''
# save the model
model_name =  '2D_CT_Recon.pt'
torch.save(model.state_dict(), '/Data/MoriRichardProject/'+model_name)


