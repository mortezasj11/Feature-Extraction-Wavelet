import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter ###
from torch import optim
from utils import iWave

class Trainer_Tensorboard:
    def __init__(self, args):

        self.criterionAE = torch.nn.MSELoss()  #criterion = DiceLoss(predict,target)#criterion = torch.nn.L1Loss()
        self.criterionCE = torch.nn.CrossEntropyLoss()

        self.model = args.model
        self.training_DataLoader = args.train_loader
        self.validation_DataLoader = args.val_loader
        self.device = args.device
        self.num_epoch = args.num_epoch
        self.epoch = args.epoch
        self.notebook = args.notebook
        self.mode = args.mode
        self.freeze = 1
        

        if args.optim == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        elif args.optim == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.4,  min_lr=5e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.2)

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.validation_loss_wave = []

    def run_trainer(self):
        self.writer = SummaryWriter() ###
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.num_epoch, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            if self.epoch <20:
                self.alpha = 1.0
            elif self.epoch >50 and self.epoch %2 == 0:
                self.alpha = 1.0
            else:
                self.alpha = 0.0
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            self.scheduler.step()       ##self.scheduler.step(self.validation_loss[i])    #self.scheduler.step() 
        self.writer.close()           
        return self.training_loss, self.validation_loss, self.learning_rate


    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses, running_corrects, data_num   = [], 0, 0  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader), leave=False)

        for i, (x, y) in batch_iter:
            input, true_Y = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            
            self.optimizer.zero_grad()  # zerograd the parameters

            if self.freeze != None and self.alpha==0:
                self.model.inc.weight.requires_grad = False
                self.model.inc.bias.requires_grad = False
                self.model.down1.weight.requires_grad = False
                self.model.down1.bias.requires_grad = False
                self.model.down2.weight.requires_grad = False
                self.model.down2.bias.requires_grad = False
                self.model.down3.weight.requires_grad = False
                self.model.down3.bias.requires_grad = False
                self.model.down4.weight.requires_grad = False
                self.model.down4.bias.requires_grad = False
                self.model.down5.weight.requires_grad = False
                self.model.down5.bias.requires_grad = False
            else:
                self.model.inc.weight.requires_grad = True
                self.model.inc.bias.requires_grad = True
                self.model.down1.weight.requires_grad = True
                self.model.down1.bias.requires_grad = True
                self.model.down2.weight.requires_grad = True
                self.model.down2.bias.requires_grad = True
                self.model.down3.weight.requires_grad = True
                self.model.down3.bias.requires_grad = True
                self.model.down4.weight.requires_grad = True
                self.model.down4.bias.requires_grad = True
                self.model.down5.weight.requires_grad = True
                self.model.down5.bias.requires_grad = True


            out, pred_Y = self.model(input)  # one forward pass #in Wavelet [4, 4, 256, 256]
            _, preds = torch.max(pred_Y, 1)
            running_corrects += torch.sum(preds == true_Y)

            lossAE = self.criterionAE(out, input)  # calculate loss

            lossCE = self.criterionCE(pred_Y, true_Y)
            loss = self.alpha * lossAE + (1 - self.alpha) * lossCE
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

            data_num += len(true_Y)
    

        #self.writer.add_images('Inputt', input, self.epoch)
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        self.writer.add_scalar('Loss/Train', np.mean(train_losses), self.epoch)
        self.writer.add_scalar('Accuracy/Train', running_corrects/data_num, self.epoch)
        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        val_loss, running_corrects, data_num = [], 0 , 0 # accumulate the losses here
        val_lossAE,val_lossCE = [],[]
        valid_losses_wave = [] # accumulate the iWave losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)


        for i, (x, y) in batch_iter:
            input, true_Y = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out, pred_Y = self.model(input)
                lossAE = self.criterionAE(out, input)  # calculate loss
                lossCE = self.criterionCE(pred_Y, true_Y)
                loss = self.alpha * lossAE + (1 - self.alpha) * lossCE
                loss_value = loss.item()
                val_loss.append(loss_value)
                val_lossAE.append(lossAE.item())
                val_lossCE.append(lossCE.item())

                _, preds = torch.max(pred_Y, 1)
                running_corrects += torch.sum(preds == true_Y)

                if self.mode.lower() =='wavelet':
                    iWave_out,  iWave_target = iWave(out),  iWave(target)
                    loss = self.criterionAE(iWave_out, iWave_target)
                    loss_value = loss.item()
                    valid_losses_wave.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
                data_num += len(true_Y)

        self.validation_loss.append(np.mean(val_loss))

        if self.mode.lower() =='regular': 
                     
            self.writer.add_images('True', torch.unsqueeze(input[:,2,:,:], dim=1) , self.epoch)
            self.writer.add_images('Val', torch.clip(torch.unsqueeze(out[:,2,:,:], dim=1) ,0,1), self.epoch)
            self.writer.add_scalar('Loss/val', np.mean(val_loss), self.epoch)
            self.writer.add_scalar('Loss/AE', np.mean(val_lossAE), self.epoch)
            self.writer.add_scalar('Loss/CE', np.mean(val_lossCE), self.epoch)
            self.writer.add_scalar('Accuracy/Val', running_corrects/data_num, self.epoch)
            self.writer.add_scalar('hyper/LR', self.optimizer.param_groups[0]['lr'], self.epoch)
            self.writer.add_scalar('hyper/alpha', self.alpha, self.epoch)

        elif  self.mode.lower() =='wavelet':
            self.validation_loss_wave.append(np.mean(valid_losses_wave))
            self.writer.add_images('True', iWave_target, self.epoch)
            self.writer.add_images('Val', torch.clip(iWave_out,0,1), self.epoch)
            self.writer.add_scalar('Loss/Val_iWave', np.mean(valid_losses_wave), self.epoch)
        batch_iter.close()

