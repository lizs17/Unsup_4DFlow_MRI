# %matplotlib widget
import os
import time
import argparse
import copy
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
from utils import reverse_order_np, rss_coil_np, fftnc_norm_np, ifftnc_norm_np, fftnc_norm_torch, ifftnc_norm_torch
from utils import R2C_cat_torch, C2R_insert_torch, save_dict_h5, read_dict_h5, metrics2D_np
from utils_4DFlow import get_quantitative_img, get_velocity_error
from model import generator_G, generator_FCN
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------
gpu = 2
C = 1.5
dataName = 'AortaS11'
flowFrame = 12
R, ACSy, ACSz = 16.0, 3, 3
# ------
dtype_float_np = np.float32
dtype_complex_np = np.complex64
dtype_float_torch = torch.float32
dtype_complex_torch = torch.complex64
# ------
seed = 1
device = torch.device("cuda", gpu)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# ------
lr = 8e-4
b1ta = 0.8
b2ta = 0.98
Niter = 20000
eval_every = 500
# ------
VENC = [150.0, 150.0, 200.0]  # cm/s
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------
postfix = ('_C' + str(C)) + ('_R' + str(R) + '_ACS' + str(ACSy) + 'x' + str(ACSz))
saveFolder = 'PIDIP_' + postfix + '/'
# ------
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
caseFolder = os.path.join(saveFolder, dataName + '/')
if not os.path.exists(caseFolder):
    os.mkdir(caseFolder)
pretrainFolder = os.path.join(caseFolder, 'pretrain' + '/')
if not os.path.exists(pretrainFolder):
    os.mkdir(pretrainFolder)
log_pretrain_Path = os.path.join(pretrainFolder, 'log_pretrain.txt')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
data = read_dict_h5(os.path.join('data/', dataName + '.h5'))
imgGT = data['imgGT']  # (Nx, Ny, Nz, Nt, Nv)
smaps = data['smaps']  # (Nx, Ny, Nz, Nc)
maskROI = data['maskROI']  # (Nx, Ny, Nz)
# ------
Nx, Ny, Nz, Nt, Nv = imgGT.shape
Nc = smaps.shape[3]
imSize = (Nz, Ny, Nx)
# ------
masksName = 'mask' + \
            ('_SZ' + str(Ny) + 'x' + str(Nz) + 'x' + str(Nt)) + \
            ('_R' + str(R)) + ('_ACS' + str(ACSy) + 'x' + str(ACSz))
masks = read_dict_h5(os.path.join('mask/', masksName + '.h5'))['mask']
masks = np.reshape(masks, (1, Ny, Nz, Nt))
masks = np.repeat(masks, axis=0, repeats=Nx)  # (Nx, Ny, Nz, Nt)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------ dtype
imgGT = imgGT.astype(dtype_complex_np)
smaps = smaps.astype(dtype_complex_np)
masks = masks.astype(dtype_float_np)
maskROI = maskROI.astype(dtype_float_np)
# ------ shape: (Nc, Nt, Nv, Nz, Ny, Nx)
imgGT = reverse_order_np(imgGT)  # (Nv, Nt, Nz, Ny, Nx)
imgGT = np.transpose(imgGT, (1, 0, 2, 3, 4))  # (Nt, Nv, Nz, Ny, Nx)
imgGT = np.reshape(imgGT, (1, Nt, Nv, Nz, Ny, Nx))
smaps = np.reshape(reverse_order_np(smaps), (Nc, 1, 1, Nz, Ny, Nx))
masks = np.reshape(reverse_order_np(masks), (1, Nt, 1, Nz, Ny, Nx))
maskROI = np.reshape(reverse_order_np(maskROI), (1, 1, Nz, Ny, Nx))
# ------ normalization
imgGT = imgGT / np.max(np.abs(imgGT))
smaps = smaps / rss_coil_np(smaps, dim=0)
# ------ undersampling
kdata = masks * fftnc_norm_np(imgGT * smaps, dim=(-3, -2, -1))  # (Nc, Nt, Nv, Nz, Ny, Nx)
# ------ zero-filling recon
AHy_np = np.sum(np.conj(smaps) * ifftnc_norm_np(kdata, dim=(-3, -2, -1)), axis=0, keepdims=True)  # (1, Nt, Nv, Nz, Ny, Nx)
# ------ tensor data
M_t = torch.tensor(masks, dtype=dtype_float_torch).to(device)
S_t = torch.tensor(smaps, dtype=dtype_complex_torch).to(device)
y_t = torch.tensor(kdata, dtype=dtype_complex_torch).to(device)
# ------ ground-truth
imgRef = np.reshape(imgGT, (Nt, Nv, Nz, Ny, Nx))
imgRef = np.transpose(imgRef, (1, 0, 2, 3, 4))  # (Nv, Nt, Nz, Ny, Nx)
imgRef_MA, imgRef_V = get_quantitative_img(imgRef, VENC=VENC)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
size_z = (8, 8)
# ------
z_Path = os.path.join(caseFolder, 'z.h5')
# ------ get latent variable
if not os.path.exists(z_Path):
    z_np_endpoint = np.random.normal(size=(2, 16) + size_z).astype(dtype_float_np)
    z_np = np.zeros((Nz, 16) + size_z, dtype=dtype_float_np)  # (Nz, ch_z) + size_z
    for ss in range(Nz):
        ratio = ss / (Nz - 1)  # 0 ~ 1
        z_np[ss,:,:,:] = ratio * z_np_endpoint[0,:,:,:] + (1.0 - ratio) * z_np_endpoint[1,:,:,:]
    z_np = np.reshape(z_np, (1, Nz, 16) + size_z)  # (1, Nz, ch_z) + size_z
    z_np = np.transpose(z_np, (0, 2, 1, 3, 4))  # (1, ch_z, Nz) + size_z
    z_np = z_np.astype(dtype_float_np)
    save_dict_h5(h5_path=z_Path, np_dict={'z': z_np})
else:
    z_np = read_dict_h5(z_Path)['z']
# ------
z_t = torch.tensor(z_np, dtype=dtype_float_torch).to(device)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------s-------------------------------------------
# ------
G = generator_G(device=device, imSize=imSize, C=C, Nt=Nt)
G.to(device)
G.weight_init()
# ------
C = generator_FCN(device=device, imSize=imSize, C=C, Nt=Nt)
C.to(device)
C.weight_init()
# ======================================================================================================================
# ====================================================================================================================== pretrain G
# ======================================================================================================================
# ------ normalize k-space data
G.eval()
C.eval()
with torch.no_grad():
    xft = G(z_t)  # (Nt, 2 * Nv) + imSize
    xtmp = C(xft)
    xtmp = R2C_cat_torch(xtmp, axis=1)  # (Nt, Nv) + imSize
    xtmp = torch.reshape(xtmp, (1, Nt, Nv) + imSize)  # (1, Nt, Nv) + imSize
    scaling_factor_pretrain = np.linalg.norm(np.mean(xtmp.detach().cpu().numpy(),axis=1)) / np.linalg.norm(np.mean(AHy_np,axis=1))
y_pretrain = y_t * scaling_factor_pretrain
# save scaling factor
save_dict_h5(
    h5_path=os.path.join(pretrainFolder,'scaling_factor_pretrain.h5'),
    np_dict={'scaling_factor_pretrain': float(scaling_factor_pretrain)}
)
# ------ pretrain
# loss
lossFunc_pretrain = nn.MSELoss(reduction='sum')
# optimizer
optimizer_pretrain = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr},
    {'params': filter(lambda p: p.requires_grad, C.parameters()), 'lr': lr},
], betas=(b1ta, b2ta))
# train
tic_iter = time.time()
for iter in range(Niter):
    G.train()
    C.train()
    optimizer_pretrain.zero_grad()
    # forward
    x_feature = G(z_t)  # (Nt, 2 * Nv) + imSize
    x_rec = C(X=x_feature)
    x_rec = R2C_cat_torch(x_rec, axis=1)  # (Nt, Nv) + imSize
    x_rec = torch.reshape(x_rec, (1, Nt, Nv) + imSize)  # (1, Nt, Nv) + imSize
    k_rec = fftnc_norm_torch(S_t * x_rec, dim=(-3, -2, -1))
    # loss
    lossDC = lossFunc_pretrain(C2R_insert_torch(M_t * k_rec), C2R_insert_torch(M_t * y_pretrain))
    # total loss
    lossAll = lossDC
    # backward
    lossAll.backward()
    # optimize weight
    optimizer_pretrain.step()
    # ------ evaluation
    # evaluate and save when encountering a lower loss
    if ((iter + 1) % eval_every == 0):
        G.eval()
        C.eval()
        imgFT = G(z_t)
        imgRec = C(X=imgFT)
        # rec
        imgRec_t = R2C_cat_torch(imgRec, axis=1)  # (Nt, Nv) + imSize
        imgRec_np = imgRec_t.detach().cpu().numpy()
        imgRec_np = np.transpose(imgRec_np, (1, 0, 2, 3, 4))  # (Nv, Nt) + imSize
        imgRec_MA, imgRec_V = get_quantitative_img(imgRec_np, VENC=VENC)
        # metrics
        MSE = np.mean(metrics2D_np(imgRef_MA, imgRec_MA, name='MSE'))
        MAE = np.mean(metrics2D_np(imgRef_MA, imgRec_MA, name='MAE'))
        RMSE_Vabs, mErr_Vang = get_velocity_error(
            imgV_RC=imgRec_V, imgV_GT=imgRef_V, V_mask=maskROI, frame_t=flowFrame)
        # print
        logStr1 = "iter: {:d}  |  time: {:.1f}  |  loss: {:.1f}".format(
            iter + 1, time.time() - tic_iter, lossAll.item())
        logStr2 = "------ MSE: {:.2f}  |  MAE: {:.2f}  |  RMSE_Vabs: {:.2f}(cm/s)  |  mErr_Vang: {:.2f}(degree)".format(
            MSE * 100000, MAE * 1000, RMSE_Vabs, mErr_Vang)
        logStr = logStr1 + '\n' + logStr2
        print(logStr)
        with open(log_pretrain_Path, 'a') as f:
            f.write(logStr)
            f.write('\n')
# ------ save
GName = 'G' + ('_it' + str(Niter))
CName = 'C' + ('_it' + str(Niter))
torch.save(G.state_dict(), os.path.join(pretrainFolder, GName + '.pt'))
torch.save(C.state_dict(), os.path.join(pretrainFolder, CName + '.pt'))


