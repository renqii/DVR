import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, opt):
        super(Value, self).__init__()
        self.dim = opt.user_dim
        if opt.backbone == 'bpr' or opt.backbone == 'lightgcn' or opt.backbone == 'mgcf':
            self.linear1 = nn.Linear(self.dim * 3 + 1, 256, bias=True)
        elif opt.backbone == 'ncf':
            self.linear1 = nn.Linear(self.dim * 6 + 1, 256, bias=True)
        self.linear2 = nn.Linear(256, 128, bias=True)
        self.linear3 = nn.Linear(128, opt.h_dim, bias=False)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
    
    def forward(self, user, pos_item, neg_item, loss_feature):
        state = torch.cat((user, pos_item, neg_item, loss_feature), dim=1).squeeze()
        a = torch.tanh(self.linear1(state))
        b = torch.tanh(self.linear2(a))
        c = torch.tanh(self.linear3(b))
        return c