# original ole
import math

import torch
import torch.nn as nn
from utils import FullGatherLayer, cust_norm

class ssole_loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gamma = 0.5

    def high_rank_loss(self, z_mat):
        B = z_mat.shape[0] + 0.0
        
        rank = torch.norm(z_mat, p='nuc', dim=(0, 2))

        beta = (rank/B).arcsin() - math.asin(1/math.sqrt(B))

        cos_a_2 = (beta.cos().pow(2) * B - 1.0)/(B-1.0)
        
        cos_a_1 = (B**2/rank.pow(2) - 1.0)/(B-1.0)
        
        cos_a = self.gamma * cos_a_1 + (1.0-self.gamma) * cos_a_2
        
        return cos_a.mean(), rank.mean()

    def low_rank_loss(self, z_mat, centered_vec=None):
        B = z_mat.shape[1] + 0.0
        
        if centered_vec is None:
            centered_vec = z_mat.mean((1,), keepdim=True)

        rank = torch.norm(z_mat - centered_vec.detach(), p='nuc', dim=(1, 2)) #B
        
        cos_a = rank.pow(2)/((B-1)*(B-1))

        return cos_a.mean(), rank.mean(), centered_vec

    def forward(self, z_mat):
        B, N, d = z_mat.shape

        loss_low_rank, rank_l, centered_vec = self.low_rank_loss(z_mat)

        z_mat_gather = torch.cat(FullGatherLayer.apply(z_mat), dim=0)

        loss_high_rank, rank_h = self.high_rank_loss(z_mat_gather)

        loss = loss_low_rank + self.config.lambda_ * loss_high_rank

        info = {
            'loss': loss,
            'rk_l': rank_l,
            'rk_h': rank_h,
            'loss_low': loss_low_rank,
            'loss_high': loss_high_rank,
        }

        return loss, info