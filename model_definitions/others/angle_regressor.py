import torch
import torch.nn as nn
import math
import numpy as np


class AngleRegressor(nn.Module):

    def __init__(self, input_size, angle_sample=np.pi/4):
        super().__init__()

        """
        :outp_size: round(360/degree)
        """

        # self.sin_fc = nn.Sequential(nn.Linear(input_size*2, outp_size), nn.Tanh())
        # self.cos_fc = nn.Sequential(nn.Linear(input_size*2, outp_size), nn.Tanh())

        num_of_outp = round(2*np.pi/angle_sample)
        self.Fcell = nn.Sequential(nn.Linear(input_size*2, input_size), # input_size -> 1024
                                   nn.Linear(input_size, 64),
                                   nn.Linear(64, num_of_outp),
                                   nn.Softmax())
        self.Ftheta = nn.Sequential(nn.Linear(input_size*2, num_of_outp), nn.Sigmoid())
        self.angle_sample = angle_sample

    def forward(self, ori_image, rot_image, angle, mode='train'):
        """
        :ori_image: Embedding of the original image
        :rot_image: Embedding of the rotate image
        """

        # sin = self.sin_fc(torch.cat((ori_image, rot_image), dim=1))
        # cos = self.cos_fc(torch.cat((ori_image, rot_image), dim=1))
        k = torch.floor(angle/self.angle_sample).long()
        if mode == 'train':
            cell_pred = self.Fcell(torch.cat((ori_image, rot_image), dim=1))
            f_k = self.Ftheta(torch.cat((ori_image, rot_image), dim=1))[[i for i in range(k.shape[0])], k] # [batch, k]
            f_k = f_k.squeeze() # [batch]
            theta_pred = self.angle_sample*(f_k + k)
            return k, cell_pred, theta_pred
        else:
            cells_pred = self.Fcell(torch.cat((ori_image, rot_image), dim=1)) # [batch, num_outp]
            cell_pred = torch.argmax(cells_pred, dim=1) # [batch]
            f_k = self.Ftheta(torch.cat((ori_image, rot_image), dim=1))[[i for i in range(cell_pred.shape[0])], cell_pred]
            theta_pred = self.angle_sample*(f_k + cell_pred) # [batch]
            return k, cells_pred, theta_pred
