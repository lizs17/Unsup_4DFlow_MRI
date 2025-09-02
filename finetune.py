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
from utils import reverse_order_np, rss_coil_np, fftnc_norm_np, ifftnc_norm_np, fftnc_norm_torch, ifftnc_norm_torch, fft1c_norm_torch, ifft1c_norm_torch
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
pretrainIter = 20000
NitADMM = 20
NitCG = 5
NitGz = 100
lrGz = 1e-4
betasGz = (0.8, 0.98)
rhoGz = 1e-3
lamS = 2e-4
rhoS = 5e-2
# ------
VENC = [150.0, 150.0, 200.0]  # cm/s
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------
postfix = ('_C' + str(C)) + ('_R' + str(R) + '_ACS' + str(ACSy) + 'x' + str(ACSz))
saveFolder = 'PIDIP_' + postfix + '/'
# ------
caseFolder = os.path.join(saveFolder, dataName + '/')
pretrainFolder = os.path.join(caseFolder, 'pretrain/')
finetuneFolder = os.path.join(caseFolder, 'finetune' + '/')
if not os.path.exists(finetuneFolder):
    os.mkdir(finetuneFolder)
log_finetune_Path = os.path.join(finetuneFolder, 'log_finetune.txt')
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
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------ load pretrained G and C
G_pretrain = 'G' + ('_it' + str(pretrainIter)) + '.pt'
C_pretrain = 'C' + ('_it' + str(pretrainIter)) + '.pt'
G.load_state_dict(torch.load(os.path.join(pretrainFolder, G_pretrain), map_location=device))
C.load_state_dict(torch.load(os.path.join(pretrainFolder, C_pretrain), map_location=device))
# ------ load scaling factor
scaling_factor_finetune = read_dict_h5(os.path.join(pretrainFolder,'scaling_factor_pretrain.h5'))['scaling_factor_pretrain']
# ------ normalize k-space data
y_finetune = y_t * scaling_factor_finetune
# save scaling factor
save_dict_h5(
    h5_path=os.path.join(finetuneFolder,'scaling_factor_finetune.h5'),
    np_dict={'scaling_factor_finetune': float(scaling_factor_finetune)}
)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def AHAop(x):
    # x: (1, Nt, Nv) + imSize
    out = M_t * fftnc_norm_torch(S_t * x, dim=(-3, -2, -1))
    out = torch.sum(torch.conj(S_t) * ifftnc_norm_torch(out, dim=(-3, -2, -1)), dim=0, keepdim=True)
    return out
# ------
AHy_t = torch.sum(torch.conj(S_t) * ifftnc_norm_torch(y_finetune, dim=(-3, -2, -1)), dim=0, keepdim=True)
# ------
def SoftThresh(x, thresh):
    m = (torch.abs(x) >= thresh)
    m = m * ((torch.abs(x) - thresh) / torch.abs(x))
    out = x * m
    return out
# ------
D_np = 0.5 * np.array([
    [1.0, 0.0, 0.0, 0.0],
    [-1.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 1.0],
])
D_np = D_np.astype(dtype_float_np)
D_t = torch.tensor(D_np, dtype=dtype_complex_torch).to(device)
# ------ Transformation
def Dop(X):
    # X: (1, Nt, Nv, Nz, Ny, Nx)
    tmp = fft1c_norm_torch(X, dim=1)
    tmp = torch.permute(tmp, (0, 2, 1, 3, 4, 5))  # (1, Nv, Nt, Nz, Ny, Nx)
    tmp = torch.reshape(tmp, (Nv, Nt * Nz * Ny * Nx))
    out = torch.matmul(D_t, tmp)  # (Nv, Nt * Nz * Ny * Nx)
    return out
def DHop(H):
    # Z: (Nv, Nt * Nz * Ny * Nx)
    tmp = torch.matmul(torch.conj(D_t.T), H)  # (Nv, Nt * Nz * Ny * Nx)
    tmp = torch.reshape(tmp, (1, Nv, Nt, Nz, Ny, Nx))
    tmp = torch.permute(tmp, (0, 2, 1, 3, 4, 5))  # (1, Nt, Nv, Nz, Ny, Nx)
    out = ifft1c_norm_torch(tmp, dim=1)
    return out
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------ initial value
XSize = (1, Nt, Nv) + imSize
HSize = (Nv, Nt * int(np.prod(np.array(imSize))))
# ------
G.eval()
C.eval()
xft = G(z_t)  # (Nt, 2 * Nv) + imSize
Gz0 = C(X=xft)
Gz0 = R2C_cat_torch(Gz0, axis=1)  # (Nt, Nv) + imSize
Gz = torch.reshape(Gz0, (1, Nt, Nv) + imSize)  # (1, Nt, Nv) + imSize
# ------
X = Gz
# ------
H = Dop(X)
# ------
Lam = torch.zeros(HSize, dtype=dtype_complex_torch).to(device)
Gam = torch.zeros(XSize, dtype=dtype_complex_torch).to(device)
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ------
lossFunc = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lrGz},
    {'params': filter(lambda p: p.requires_grad, C.parameters()), 'lr': lrGz},
], betas=betasGz)
# ------
Normal_op = lambda x: AHAop(x) + rhoS * DHop(Dop(x)) + rhoGz * x
# ------
tic_iter = time.time()
for iter_ADMM in range(NitADMM):
    with torch.no_grad():
        # ============ solve subproblem 1: update X ============
        b = AHy_t + DHop(rhoS * H + Lam) + rhoGz * Gz - Gam
        r = b - Normal_op(x=X)
        p = r
        rHr = torch.sum(torch.abs(r) ** 2).item()
        for iter_sub1 in range(NitCG):
            # gradient descent
            Bp = Normal_op(x=p)
            alpha = rHr / torch.abs(torch.sum(torch.conj(p) * Bp)).item()
            X = X + alpha * p
            r = r - alpha * Bp
            rHr_new = torch.sum(torch.abs(r) ** 2).item()
            # conjugate gradient
            beta = rHr_new / rHr
            p = r + beta * p
            rHr = rHr_new
        # ------ recon
        imgRec_np = X.detach().cpu().numpy()  # (1, Nt, Nv) + imSize
        imgRec_np = np.reshape(imgRec_np, (Nt, Nv) + imSize)
        imgRec_np = np.transpose(imgRec_np, (1, 0, 2, 3, 4))  # (Nv, Nt) + imSize
        imgRec_MA, imgRec_V = get_quantitative_img(imgRec_np, VENC=VENC)
        # ------ eval
        MSE = np.mean(metrics2D_np(imgRef_MA, imgRec_MA, name='MSE'))
        MAE = np.mean(metrics2D_np(imgRef_MA, imgRec_MA, name='MAE'))
        RMSE_Vabs, mErr_Vang = get_velocity_error(
            imgV_RC=imgRec_V, imgV_GT=imgRef_V, V_mask=maskROI, frame_t=flowFrame)
        # print
        logStr1 = "iter: {:d}  |  time: {:.1f}".format(
            iter_ADMM + 1, time.time() - tic_iter)
        logStr2 = "------ MSE: {:.2f}  |  MAE: {:.2f}  |  RMSE_Vabs: {:.2f}(cm/s)  |  mErr_Vang: {:.2f}(degree)".format(
            MSE * 100000, MAE * 1000, RMSE_Vabs, mErr_Vang)
        logStr = logStr1 + '\n' + logStr2
        print(logStr)
        with open(log_finetune_Path, 'a') as f:
            f.write(logStr)
            f.write('\n')
        # ============ solve subproblem 2: update H ============
        H = SoftThresh(Dop(X) - (Lam / rhoS), thresh=(lamS / rhoS))
    # ============ solve subproblem 3: update theta ============
    for iter_Gz in range(NitGz):
        G.train()
        C.train()
        optimizer.zero_grad()
        # forward
        x_feature = G(z_t)  # (Nt, 2 * Nv) + imSize
        x_rec = C(X=x_feature)
        x_rec = R2C_cat_torch(x_rec, axis=1)  # (Nt, Nv) + imSize
        x_rec = torch.reshape(x_rec, (1, Nt, Nv) + imSize)  # (1, Nt, Nv) + imSize
        # loss
        loss = lossFunc(C2R_insert_torch(x_rec), C2R_insert_torch(X + (1 / rhoGz) * Gam))
        # backward
        loss.backward()
        # optimize weight
        optimizer.step()
    with torch.no_grad():
        # ============ update Gz ============
        G.eval()
        C.eval()
        xtempft = G(z_t)  # (Nt, 2 * Nv) + imSize
        Gz = C(X=xtempft)
        Gz = R2C_cat_torch(Gz, axis=1)  # (Nt, Nv) + imSize
        Gz = torch.reshape(Gz, (1, Nt, Nv) + imSize)  # (1, Nt, Nv) + imSize
        # ============ dual ascent ============
        Lam = Lam + rhoS * (H - Dop(X))
        Gam = Gam + rhoGz * (X - Gz)
    # ============ save result ============
    saveName = dataName + ('_rec' + str(iter_ADMM + 1)) + '.h5'
    imgSave = X.detach().cpu().numpy()
    imgSave = np.reshape(imgSave, (Nt, Nv) + imSize)  # (Nt, Nv) + imSize
    imgSave = np.transpose(imgSave, (1, 0, 2, 3, 4))  # (Nv, Nt) + imSize
    imgSave = reverse_order_np(imgSave)  # (Nx, Ny, Nz, Nt, Nv)
    save_dict_h5(
        h5_path=os.path.join(finetuneFolder, saveName),
        np_dict={'img': imgSave}
    )


