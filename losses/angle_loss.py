import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleLoss(nn.Module):

    def __init__(self, config):

        self.alpha_1 = config.train.alpha_1
        self.alpha_2 = config.train.alpha_2

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, sin_pred, cos_pred, sin_label, cos_label):

        loss = self.alpha_1*self.mse(sin_pred - sin_label) + \
               self.alpha_1*self.mse(cos_pred - cos_label) + \
               self.alpha_2*(sin_pred.pow(2) + cos_pred.pow(2) - 1)

        return loss
