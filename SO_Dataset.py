# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from transformers import BertTokenizer

class SO_Dataset(Dataset):
  def __init__(self, df: DataFrame, tokenizer: BertTokenizer, seq_length: int=64):
    self.df = df
    self.tokenizer = tokenizer
    self.seq_length = seq_length
    
  def __len__(self):
    return self.df.shape[0]
  
  def __getitem__(self, idx):
    inputs = self.tokenizer(self.df['Text'].loc[idx], return_tensors='pt', 
                            max_length=self.seq_length, truncation=True, padding='max_length')
    return  {'input_ids':inputs['input_ids'].squeeze(0),
             'token_type_ids':inputs['token_type_ids'].squeeze(0),
             'attention_mask':inputs['attention_mask'].squeeze(0),
             'label':torch.tensor(self.df['Y'].iloc[idx], dtype=torch.long)}
