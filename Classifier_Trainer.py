# -*- coding: utf-8 -*-
import os
import abc
import numpy as np
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer():

  def __init__(self, run_name: str, model: torch.nn.Module, n_epochs: int, train_dataset: DataFrame, 
               batch_size: int, criterion: torch.nn.modules.loss, optimizer: torch.optim, 
               valid_dataset: DataFrame=None, shuffle: bool=True, drop_last: bool=True, 
               save_model_mode: str=None, save_model_path:str=None, log_path: str=None):
    """
    This class is used as a parent class for classifier model training
    children need to implement _predict_batch and _compute_loss methods
    
    Parameters
    ----------
    run_name : str
      name of the run, will be used for saving models and logs
    
    model : torch.nn.Module
      a model to train
      
    n_epochs : int
      number of epochs
    
    train_dataset : DataFrame
      training data in pandas.DataFrame
      
    batch_size : int
      batch size
      
    criterion : torch.nn.modules.loss
      a loss function
      
    optimizer : torch.optim
      an optimizer
      
    valid_dataset : DataFrame
      optional validation data. If not provided, save_model_mode='best' will not 
      be available and will defaul to 'best'
      
    shuffle : bool
      an indicator for shuffling training set when feeding batch
      
    drop_last : bool
      an indicator for dropping last batch (which batch size might be smaller) 
      during training and evaluating
      
    save_model_mode : str
      one of {'best', 'all', None}. The 'best' mode will only save models with 
      inproved validation loss; 'all' mode will save all models; None mode will 
      not save models
      
    save_model_path : str
      directory for saved models and states
      
    log_path : str
      directory for tensorboard log

    Returns
    -------
    None
      
    """
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.run_name = run_name
    self.model = model
    self.model.to(self.device)
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.criterion = criterion
    self.optimizer = optimizer
    self.valid_dataset = valid_dataset
    self.shuffle = shuffle
    self.global_step = 0
    self.save_model_path = os.path.join(save_model_path, self.run_name)
    if save_model_path != None and not os.path.isdir(self.save_model_path):
      os.makedirs(self.save_model_path)
    
    self.best_loss = float('inf')
    
    """ change 'best' mode to 'all' if validation set is not provided """
    self.save_model_mode = save_model_mode 
    if self.save_model_mode == 'best' and self.valid_dataset == None:
      self.save_model_mode = 'all'
      
      
    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                   shuffle=self.shuffle, drop_last=drop_last)
    if valid_dataset != None:
      self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                     shuffle=False, drop_last=drop_last)
    else:
      self.valid_loader = None
    
    
    self.log_path = os.path.join(log_path, self.run_name)
    if log_path != None and not os.path.isdir(self.log_path):
      os.makedirs(self.log_path)
    self.tensorboard_writer = SummaryWriter(self.log_path) if log_path != None else None
    
  def train(self):
    for epoch in range(self.n_epochs):
      accum_train_loss = 0
      valid_loss = None
      loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
      
      for batch_id, train_batch in loop:
        self.global_step += 1
        """ forward """
        train_pred = self._predict_batch(train_batch)
        train_loss = self._compute_loss(train_pred, train_batch)
        
        """ record loss """
        accum_train_loss += train_loss.item()
        if self.tensorboard_writer != None:
          self.tensorboard_writer.add_scalar("train/loss", accum_train_loss/ (batch_id+1), self.global_step)
        """ backward """
        self.optimizer.zero_grad()
        train_loss.backward()
        
        """ update """
        self.optimizer.step()
        
        """ validation loss at end of epoch"""
        if self.valid_loader != None and batch_id == len(self.train_loader) - 1:
          valid_loss = self.evaluate()
          if self.tensorboard_writer != None:
            self.tensorboard_writer.add_scalar("valid/loss", valid_loss, self.global_step)

        """ print progress bar """
        loop.set_description(f'Epoch [{epoch + 1}/{self.n_epochs}]')
        loop.set_postfix(train_batch_loss=train_loss.item(), 
                         accumulate_train_loss=accum_train_loss/ (batch_id+1), 
                         valid_loss=valid_loss)
        
      """ end of epoch """
      if self.save_model_mode == 'all':
        self.save_model(epoch, train_loss.item(), valid_loss)
      elif self.save_model_mode == 'best':
        if epoch == 0 or valid_loss < self.best_loss:
          self.save_model(epoch, train_loss.item(), valid_loss)
          
      """ update best validation loss """
      self.best_loss = min(self.best_loss, valid_loss)
            
      
  def evaluate(self) -> np.ndarray:
    """
    This method calculats validation loss

    Returns
    -------
    Numpy.array
      a nparray that holds a scaler of total loss

    """
    with torch.no_grad():
      valid_total_loss = 0
      for valid_batch in self.valid_loader:
        valid_pred = self._predict_batch(valid_batch)
        valid_losses = self._compute_loss(valid_pred, valid_batch)
        valid_total_loss += valid_losses.item()
      return valid_total_loss/ len(self.valid_loader)
    
  def save_model(self, epoch, train_loss, valid_loss):
    torch.save(self.model.state_dict(), 
               f'{self.save_model_path}\\Epoch-{epoch}_trainloss-{train_loss:.4f}_validloss-{valid_loss:.4f}_model.pth')
    

  @abc.abstractmethod
  def _predict_batch(self, batch: Dict) -> Dict:
    """
    This method inputs a batch of data and use model to predict
    predictions can be used for calculating loss
    
    Parameters
    ----------
    batch : Dict
      a batch of training data

    Returns
    -------
    Dict
      predictions from model

    """
    return NotImplemented
  
  @abc.abstractmethod  
  def _compute_loss(self, pred: Dict, label: Dict) -> Dict:
    """
    This method inputs a batch of prediction and computes loss

    Parameters
    ----------
    pred : Dict
      predictions from model of a batch of training data
    label : Dict
      label of a batch of training data

    Returns
    -------
    Dict
      Dict contains avg loss (as scalers) of each loss function

    """
    return NotImplemented
  
  