# -*- coding: utf-8 -*-
from Classifier_Trainer import Trainer
from typing import Dict

class BERT_cls_Trainer(Trainer):
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
    input_ids = batch['input_ids'].to(self.device)
    attention_mask = batch['attention_mask'].to(self.device)
    return self.model(input_ids, attention_mask)
    
  def _compute_loss(self, pred: Dict, batch: Dict) -> Dict:
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
    label = batch['label']
    return self.criterion(pred.squeeze(-1).to(self.device), label.to(self.device))
    
    
    
    