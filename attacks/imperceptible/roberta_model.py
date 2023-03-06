import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU

from torch.nn import BatchNorm1d, Linear, ReLU
from torch.autograd import Variable
import numpy as np


class stance_model(nn.Module):
    def __init__(self, roberta_model, args):
        super(stance_model, self).__init__()
        self.roberta_hidden_dim = args.roberta_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = roberta_model
        self.proj_match = nn.Linear(self.roberta_hidden_dim, self.num_labels)


    def forward(self, inp_tensor, msk_tensor):
        _, pooler = self.pred_model(inp_tensor, msk_tensor,return_dict=False)
        outputs = self.dropout(pooler)
        score = self.proj_match(outputs).squeeze(-1)
        #score = torch.tanh(score)
        return score
