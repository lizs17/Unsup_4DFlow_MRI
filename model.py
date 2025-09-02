import os
import time
import argparse
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class BasicModule(nn.Module):
    def __init__(self, inCH, outCH, groups, outSize=None):
        # 'inCH' and 'outCH' should be multipliers of 'groups' (Nt)
        super(BasicModule, self).__init__()
        self.inCH = inCH
        self.outCH = outCH
        self.groups = groups
        self.ch = outCH // groups
        self.outSize = outSize
        # conv
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=inCH, out_channels=outCH, groups=groups,
                kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # normalization
        self.norm = nn.InstanceNorm3d(num_features=self.ch)
        # upsample
        if outSize is not None:
            self.up = nn.Upsample(size=outSize, mode='trilinear')
    def forward(self, x):
        # x: (1, Nt * CHin, Nz, Sy, Sx)
        _, _, Nz, Sy, Sx = x.size()
        out = self.conv(x)  # (1, Nt * CHout, Nz, Sy, Sx)
        out = torch.reshape(out, (self.groups, self.ch, Nz, Sy, Sx))
        out = self.norm(out)
        if self.outSize is not None:
            out = self.up(out)
            out = torch.reshape(out, (1, self.groups * self.ch) + self.outSize)
        else:
            out = torch.reshape(out, (1, self.groups * self.ch, Nz, Sy, Sx))
        return out

class generator_G(nn.Module):
    def __init__(self, device, imSize, C, Nt, Nv=4):
        super(generator_G, self).__init__()
        # ------
        self.C = C
        self.Nt = Nt
        self.Nv = Nv
        self.ch = int(Nv * C * 2)
        # ------
        self.device = device
        self.imSize = imSize
        Nz, Ny, Nx = imSize
        # middle size
        midNy = min(128, Ny)
        midNx = min(128, Nx)
        # network (separate part)
        self.conv_backbone = nn.Sequential(
            nn.Conv3d(
                in_channels=Nt * 16, out_channels=16 * Nt * self.ch, groups=Nt,
                kernel_size=3, stride=1, padding=1, bias=True),
            BasicModule(inCH=16 * Nt * self.ch, outCH=8 * Nt * self.ch, groups=Nt, outSize=(Nz, 16, 16)),
            BasicModule(inCH=8 * Nt * self.ch, outCH=4 * Nt * self.ch, groups=Nt, outSize=(Nz, 32, 32)),
            BasicModule(inCH=4 * Nt * self.ch, outCH=2 * Nt * self.ch, groups=Nt, outSize=(Nz, 64, 64)),
            BasicModule(inCH=2 * Nt * self.ch, outCH=1 * Nt * self.ch, groups=Nt, outSize=(Nz, midNy, midNx)),
            BasicModule(inCH=1 * Nt * self.ch, outCH=1 * Nt * self.ch, groups=Nt, outSize=(Nz, Ny, Nx)),
            BasicModule(inCH=1 * Nt * self.ch, outCH=1 * Nt * self.ch, groups=Nt),
        )
    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data.normal_(0.0, 0.02)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm3d):
                 nn.init.constant_(module.weight, 1)
                 nn.init.constant_(module.bias, 0)
    def forward(self, z):
        '''
        Input:
            z: (1, Cz) + zSize
        '''
        # since we use the group-convolution, we need to replicate the latent variable channels to match the group number
        zin = torch.tile(z, dims=(1, self.Nt, 1, 1, 1))  # (1, Cz * Nt) + zSize
        # to features
        x_feature = self.conv_backbone(zin)
        return x_feature
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Gres(nn.Module):
    def __init__(self, device, C, Nt, Nv=4):
        super(Gres, self).__init__()
        # ------
        self.device = device
        self.C = C
        self.Nt = Nt
        self.Nv = Nv
        self.ch = int(Nv * C * 2)
        # aggregation weights
        self.conv = nn.Conv3d(
            in_channels= Nt * self.ch, out_channels= Nt * self.ch, groups=1,
            kernel_size=3, stride=1, padding=1, bias=True)
        # activation
        self.act = nn.ELU(alpha=1.0, inplace=True)
        self.norm = nn.BatchNorm3d(num_features=self.ch)
    def forward(self, X):
        # X: (1, Nt * Nv * C * 2, Nz, Ny, Nx)
        _, _, Nz, Ny, Nx = X.size()
        out = self.conv(X) / self.Nt  # take mean for normalization
        out = self.act(out)
        out = torch.reshape(out, (self.Nt, self.ch, Nz, Ny, Nx))
        out = self.norm(out)
        out = torch.reshape(out, (1, self.Nt * self.ch, Nz, Ny, Nx))
        return out

class Gfus(nn.Module):
    def __init__(self, device, C, Nt, Nv=4):
        super(Gfus, self).__init__()
        self.device = device
        self.C = C
        self.Nt = Nt
        self.Nv = Nv
        self.ch = int(Nv * C * 2)
        # upd0 weights
        self.ConvInput = nn.Conv3d(
            in_channels= Nt * self.ch, out_channels= Nt * self.ch, groups=Nt,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ELU(alpha=1.0, inplace=True)
        self.norm = nn.InstanceNorm3d(num_features=self.ch)
        # upd weights
        self.ConvConnect = nn.Conv3d(
            in_channels= 2 * Nt * self.ch, out_channels= Nt * self.ch, groups=1,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.actFinal = nn.ELU(alpha=1.0, inplace=True)
    def forward(self, X, Xhid):
        # X: (1, Nt * ch, Nz, Ny, Nx)
        # Xhid: (1, Nt * ch, Nz, Ny, Nx)
        _, _, Nz, Ny, Nx = X.size()
        # input branch
        Xinput = self.act(self.ConvInput(X))
        Xinput = torch.reshape(Xinput, (self.Nt, self.ch, Nz, Ny, Nx))
        Xinput = self.norm(Xinput)
        Xinput = torch.reshape(Xinput, (1, self.Nt * self.ch, Nz, Ny, Nx))
        # hidden branch
        Xtmp = torch.cat([Xinput, Xhid], dim=1)  # (1, 2 * Nt * ch, Nz, Ny, Nx)
        Xout = self.ConvConnect(Xtmp) / self.Nt
        Xout = self.actFinal(Xout)
        return Xout

class BasicModule_res(nn.Module):
    def __init__(self, CH, groups):
        super(BasicModule_res, self).__init__()
        # 'CH' should be multipliers of 'Nt'
        self.CH = CH
        self.groups = groups
        self.ch = CH // groups
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=CH, out_channels=CH, groups=groups,
                kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.norm = nn.InstanceNorm3d(num_features=CH)
    def forward(self, x):
        # x: (1, Nt * Nv * C * 2, Nz, Ny, Nx)
        _, _, Nz, Ny, Nx = x.size()
        tmp = self.conv(x)
        tmp = torch.reshape(tmp, (self.groups, self.ch, Nz, Ny, Nx))
        tmp = self.norm(tmp)
        tmp = torch.reshape(tmp, (1, self.groups * self.ch, Nz, Ny, Nx))
        out = tmp + x
        return out

class generator_FCN(nn.Module):
    def __init__(self, device, imSize, C, Nt, Nv=4):
        super(generator_FCN, self).__init__()
        self.device = device
        self.imSize = imSize
        self.C = C
        self.Nt = Nt
        self.Nv = Nv
        self.ch = int(Nv * C * 2)
        # -----
        self.Gres = Gres(device=device, C=C, Nt=Nt, Nv=Nv)
        self.Gfus = Gfus(device=device, C=C, Nt=Nt, Nv=Nv)
        # ----- conv head
        self.conv = nn.Sequential(
            BasicModule_res(CH=Nt * self.ch, groups=Nt),
            BasicModule_res(CH=Nt * self.ch, groups=Nt),
        )
        self.convFinal = nn.Conv3d(
            in_channels=Nt * self.ch, out_channels=Nt * Nv * 2,
            kernel_size=1, stride=1, padding=0, bias=True)
    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                 nn.init.constant_(module.weight, 1)
                 if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    def forward(self, X):
        # X: (1, Nt * Nv * C * 2, Nz, Ny, Nx)
        _, _, Nz, Ny, Nx = X.size()
        # ------
        Xres = self.Gres(X=X)
        Xfus = self.Gfus(X=X, Xhid=Xres)
        # ------
        out = self.conv(Xfus)
        out = self.convFinal(out)
        out = torch.reshape(out, (self.Nt, self.Nv * 2, Nz, Ny, Nx))
        return out

