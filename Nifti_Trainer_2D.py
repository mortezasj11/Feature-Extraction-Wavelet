# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:35:53 2021

@author: MBSaad
"""
import pathlib

import torch
#from unet_recon_ver2 import UNet
from AE import AE
from trainer import Trainer
from trainer_tensorboard import Trainer_Tensorboard
from Nifti_Loader_2D import dataloader_training, dataloader_validation
from visual import plot_training
from torch.utils.tensorboard import SummaryWriter ### RUN pip install tensorboard==2.4.0


# device
if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu')
   
#device = torch.device('cpu')
# model
"""
model = UNet(in_channels=1,
             out_channels=1,
             n_blocks=6,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)
"""
model = AE(in_channel = 1, out_channel =1).to(device)
# criterion
#criterion = torch.nn.CrossEntropyLoss()
#criterion = DiceLoss(predict,target)
#criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# trainer
'''
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=50,
                  epoch=0,
                  notebook=False)

# start training
training_losses, validation_losses, learning_rates = trainer.run_trainer()
'''
trainer = Trainer_Tensorboard(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=50,
                  epoch=0,
                  notebook=False)

# start training
training_losses, validation_losses, learning_rates = trainer.run_trainer()


# plot training
fig = plot_training(training_losses,
                    validation_losses,
                    learning_rates,
                    gaussian=True,
                    sigma=1,
                    figsize=(10,4))

# save the model
model_name =  '2D_CT_Recon.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

# save fig
fig.savefig('2D_CT_Recon.png')
