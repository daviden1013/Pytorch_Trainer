# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertConfig

class BERT_Classifier(nn.Module):
  def __init__(self, n_class: int=2, model_name: str="bert-base-uncased", config: BertConfig=None):
    super().__init__()
    self.bert_layer = BertModel.from_pretrained(model_name, config=config)
    self.cls_layer = nn.Linear(768, n_class)

  def forward(self, seq, attn_masks):
    emb = self.bert_layer(seq, attention_mask = attn_masks)
    output = self.cls_layer(emb['pooler_output'].squeeze(0))
    output = torch.sigmoid(output)
    return output
