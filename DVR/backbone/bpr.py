import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class BPR(nn.Module):
    def __init__(self, opt):
        super(BPR, self).__init__()
        self.opt = opt
        self.user_embedding = nn.Embedding(opt.num_users, opt.user_dim)
        self.item_embedding = nn.Embedding(opt.num_items, opt.item_dim)
        self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)
        self.item_embedding.weight.data.normal_(0, 1.0 / self.item_embedding.embedding_dim)


    def forward(self, user_idx, item_idx):
        user = self.user_embedding(user_idx)
        item = self.item_embedding(item_idx)
        rating = (user * item).sum(dim=1).squeeze()
        return rating

    def predict(self, user_idx):
        user = self.user_embedding(user_idx)
        item = self.item_embedding.weight
        rating = torch.matmul(user, item.t())
        return rating
    
    def predict_valid(self, user_idx, item_idx):
        user = self.user_embedding(user_idx)
        item = self.item_embedding(item_idx)
        rating = torch.matmul(user.unsqueeze(1), item.permute(0, 2, 1)).squeeze()
        return rating