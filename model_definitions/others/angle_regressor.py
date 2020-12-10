import torch
import torch.nn as nn


class AngleRegressor(nn.Module):

    def __init__(self, input_size, outp_size):

        self.sin_fc = nn.Sequential(nn.Linear(input_size*2, outp_size), nn.Tanh())
        self.cos_fc = nn.Sequential(nn.Linear(input_size*2, outp_size), nn.Tanh())

    def forward(self, ori_image, rot_image):
        sin = self.sin_fc(torch.cat([ori_image, rot_image]))
        cos = self.cos_fc(torch.cat([ori_image, rot_image]))
        return sin, cos
