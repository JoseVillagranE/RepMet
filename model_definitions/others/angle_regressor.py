import torch
import torch.nn as nn


class AngleRegressor(nn.Module):

    def __init__(self, input_size, outp_size):
        super().__init__()
        self.sin_fc = nn.Sequential(nn.Linear(input_size*2, outp_size), nn.Tanh())
        self.cos_fc = nn.Sequential(nn.Linear(input_size*2, outp_size), nn.Tanh())

    def forward(self, ori_image, rot_image):
        sin = self.sin_fc(torch.cat((ori_image, rot_image), dim=1))
        cos = self.cos_fc(torch.cat((ori_image, rot_image), dim=1))
        return sin, cos
