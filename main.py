# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import torch
from transformers import BertTokenizer
from SO_Dataset import SO_Dataset
from BERT_Classifier_Model import BERT_Classifier
from BERT_Classifier_Trainer import BERT_cls_Trainer
from transformers import BertConfig
import torch.nn as nn
import torch.optim as optim

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

PATH = r'E:\David projects\Pytorch tutorial'

def main():
  
  """ load data """
  train_raw = pd.read_csv(f'{PATH}\\data\\train.csv')
  validate_raw = pd.read_csv(f'{PATH}\\data\\valid.csv')
  
  """ 
  data processing 
  """
  def subset(data, pattern):
    return data.loc[data['Tags'].str.find(pattern) != -1].reset_index(drop=True)
  
  def cleanhtml(html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', html).replace('\n', '')
  
  """ only take Python-related posts """
  train = subset(train_raw, 'python')
  test = subset(validate_raw, 'python')
  
  """ clean HTML """
  train['Body'] = train['Body'].apply(cleanhtml)
  test['Body'] = test['Body'].apply(cleanhtml)
  
  """ concate title and body """
  train['Text'] = train['Title'] + ' ' + train['Body']
  test['Text'] = test['Title'] + ' ' + test['Body']
  
  """ convert Y to binary LQ = 0, HQ = 1 """
  train['Y'] = train['Y'].map({'LQ_CLOSE':0, 'LQ_EDIT':0, 'HQ':1})
  test['Y'] = test['Y'].map({'LQ_CLOSE':0, 'LQ_EDIT':0, 'HQ':1})
  
  """ 
  trainset, validation split 
  """
  def train_valid_split(seed=123):
    np.random.seed(seed)
    valid_id = np.random.choice(train.index, int(train.shape[0] * 0.2), replace=False)
    validset = train.iloc[valid_id].reset_index(drop=True)
    trainset = train.loc[~train.index.isin(valid_id)].reset_index(drop=True)
    return trainset, validset
  
  """ 
  tokenizer 
  """
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  
  """
  make datasets
  """
  trainset, validset= train_valid_split()
  train_dataset = SO_Dataset(trainset, tokenizer)
  valid_dataset = SO_Dataset(validset, tokenizer)
  
  """
  model
  """
  config = BertConfig(dropout=0.2, attention_dropout=0.2)
  model = BERT_Classifier(config=config)
  
  """ 
  train
  """
  
  """ loss """
  criterion = nn.CrossEntropyLoss()
  """ optimizer """
  optimizer = optim.Adam(model.parameters(), lr=2e-5)
  
  trainer = BERT_cls_Trainer(run_name='test01',
                   model=model, 
                   n_epochs=5,
                   train_dataset=train_dataset,
                   batch_size=32,
                   criterion=criterion,
                   optimizer=optimizer,
                   valid_dataset=valid_dataset,
                   save_model_mode='all',
                   save_model_path=r'E:\David projects\Pytorch tutorial\trainAPP\checkpoints',
                   log_path=r'E:\David projects\Pytorch tutorial\trainAPP\log')
  
  trainer.train()

if __name__ == "__main__":
  main()


