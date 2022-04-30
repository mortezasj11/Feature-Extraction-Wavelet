# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:35:53 2021

@author: msalehjahromi
"""
import torch
from AE import AE
from dataLoaderAE import Dataset_CTLungSeg
from trainer_tensorboard import Trainer_Tensorboard
from torch.utils.tensorboard import SummaryWriter ### RUN pip install tensorboard==2.4.0
from torch.utils.data import DataLoader, random_split
import pandas as pd
import argparse

def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Lung Genimi Survival Model Training')

    # base
    parser.add_argument('--mode',                 type=str, default='regular')
    parser.add_argument('--OnlyLung',             type=bool, default='False')
    parser.add_argument('--category',             type=str, default='Pathological.T.Stage')
    parser.add_argument('--num_classes',          type=int, default=4)
    # datasets
    parser.add_argument('--CT_dir_train',         type=str, default='/Data/MoriRichardProject/Pathological.T.Stage/CT_Tr')
    parser.add_argument('--Seg_dir_train',        type=str, default='/Data/MoriRichardProject/Pathological.T.Stage/Label_Tr')
    parser.add_argument('--CT_dir_val',           type=str, default='/Data/MoriRichardProject/Pathological.T.Stage/CT_Va')
    parser.add_argument('--Seg_dir_val',          type=str, default='/Data/MoriRichardProject/Pathological.T.Stage/Label_Va')
    # model
    parser.add_argument('--in_channel',           type=int, default=10) #
    parser.add_argument('--out_channel',          type=int, default=10) #
    parser.add_argument('--model_size',           type=int, default=20)
    # optimizer
    parser.add_argument('--optim',                type=str, default="Adam")
    parser.add_argument('--lr',                   type=float, default=1.0e-3)
    parser.add_argument('--weight_decay',         type=float, default=5.0e-4)
    parser.add_argument('--num_epoch',            type=int, default=120)
    parser.add_argument('--batch_size',           type=int, default=6)
    parser.add_argument('--epoch',                type=int, default=0)
    # schedular
    parser.add_argument('--step_size',            type=int, default=40)
    parser.add_argument('--gamma',                type=int, default=0.2)
    # other
    parser.add_argument('--isTrain',              type=bool, default=True)
    parser.add_argument('--notebook',             type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = set_args()
    args.df = pd.read_excel('sandy.xlsx', engine = 'openpyxl')
    args.dfTumorSlide = pd.read_csv('SandyTumors.csv')

    # train val loader      # should be re-write
    dataset_train = Dataset_CTLungSeg(args)
    dataset_train, _ = random_split(dataset_train, [len(dataset_train), 0]) # maybe is not needed
    args.train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    args.isTrain = False
    dataset_val = Dataset_CTLungSeg(args)
    dataset_val, _ = random_split(dataset_val, [len(dataset_val), 0]) # maybe is not needed
    args.val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    print('n_train, n_val', len(dataset_train), len(dataset_val))



    # device
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



    # model
    if args.mode == 'wavelet':
        args.in_channel, args.out_channel = 4, 4
    args.model = AE(args).to(args.device) 



    # trainer & training
    trainer = Trainer_Tensorboard(args)
    training_losses, validation_losses, learning_rates = trainer.run_trainer()



    # save the model
    model_name =  '2D_AE_classification.pt'
    torch.save(args.model.state_dict(), '/Data/MoriRichardProject/'+model_name)


## Calculating the accuracy and the losses on the limitations self.alpha = 0, 1
## Changing the channel input, lowering it? line 30.31

## self.alpha = 1 if epoch<50
## self.alpha = 0 if epoch>50

## Mix the training and Val data
