import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.alpha_1 = config.train.alpha_1
        # self.alpha_2 = config.train.alpha_2
        self.mse = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, k, cell_pred, angle, theta_pred):

        # loss = self.alpha_1*self.mse(sin_pred, sin_label) + \
        #        self.alpha_1*self.mse(cos_pred, cos_label) + \
        #        self.alpha_2*(sin_pred.pow(2) + cos_pred.pow(2) - 1)
        loss = self.cross_entropy(cell_pred, k)# + self.mse(theta_pred, angle)
        return loss.mean()
