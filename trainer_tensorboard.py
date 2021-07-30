import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter ###
from torch import optim
from utils import iWave

class Trainer_Tensorboard:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 mode: str = 'regular'
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.mode = mode

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.validation_loss_wave = []

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.4,  min_lr=5e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.2)

    def run_trainer(self):
        self.writer = SummaryWriter() ###
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step

            #self.scheduler.step(self.validation_loss[i])   
            self.scheduler.step()       
        self.writer.close()           
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader), leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            
            self.optimizer.zero_grad()  # zerograd the parameters
            out,_ = self.model(input)  # one forward pass #in Wavelet [4, 4, 256, 256]
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
    

        #self.writer.add_images('Inputt', input, self.epoch)
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.epoch)
        self.writer.add_scalar('Loss/Train', np.mean(train_losses), self.epoch)
        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_losses_wave = [] # accumulate the iWave losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out,_ = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                if self.mode.lower() =='wavelet':
                    iWave_out,  iWave_target = iWave(out),  iWave(target)
                    loss = self.criterion(iWave_out, iWave_target)
                    loss_value = loss.item()
                    valid_losses_wave.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        self.validation_loss.append(np.mean(valid_losses))
        if self.mode.lower() =='regular':           
            self.writer.add_images('True', target, self.epoch)
            self.writer.add_images('Val', torch.clip(out,0,1), self.epoch)
            self.writer.add_scalar('Loss/Val', np.mean(valid_losses), self.epoch)

        elif  self.mode.lower() =='wavelet':
            self.validation_loss_wave.append(np.mean(valid_losses_wave))
            self.writer.add_images('True', iWave_target, self.epoch)
            self.writer.add_images('Val', torch.clip(iWave_out,0,1), self.epoch)
            self.writer.add_scalar('Loss/Val_iWave', np.mean(valid_losses_wave), self.epoch)
        batch_iter.close()
