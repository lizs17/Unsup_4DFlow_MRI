import numpy as np


def get_quantitative_img(img_np, VENC):
    # img: (Nv, Nt, Nz, Ny, Nx)
    # VENC: cm/s
    img_MA = np.abs(img_np[0:1,:,:,:,:])
    img_MA = img_MA / np.max(np.abs(img_MA))  # (1, Nt, Nz, Ny, Nx)
    img_V1 = np.angle(img_np[1:2,:,:,:,:] * np.conj(img_np[0:1,:,:,:,:])) / np.pi * VENC[0]
    img_V2 = np.angle(img_np[2:3,:,:,:,:] * np.conj(img_np[0:1,:,:,:,:])) / np.pi * VENC[1]
    img_V3 = np.angle(img_np[3:4,:,:,:,:] * np.conj(img_np[0:1,:,:,:,:])) / np.pi * VENC[2]
    img_V = np.concatenate([img_V1, img_V2, img_V3], axis=0)  # (3, Nt, Nz, Ny, Nx)
    return img_MA, img_V

def get_Vmag(imgV):
    out = np.sqrt(imgV[0:1,:,:,:,:]**2 + imgV[1:2,:,:,:,:]**2 + imgV[2:3,:,:,:,:]**2)
    return out

def get_velocity_error(imgV_RC, imgV_GT, V_mask, frame_t=None):

    '''
        imgV:   (3, Nt, Nz, Ny, Nx)
        V_mask: (1, 1, Nz, Ny, Nx)
    '''
    _, Nt, Nz, Ny, Nx = imgV_RC.shape
    # error magnitude
    Vabs_GT = get_Vmag(imgV_GT)
    Vabs_RC = get_Vmag(imgV_RC)
    Verr_abs = np.abs(Vabs_RC - Vabs_GT)  # (1, Nt, Nz, Ny, Nx)
    # error angle
    cos_ang = np.sum(imgV_RC * imgV_GT, axis=0, keepdims=True) / (np.linalg.norm(imgV_RC, ord=2, axis=0, keepdims=True) * np.linalg.norm(imgV_GT, ord=2, axis=0, keepdims=True))
    cos_ang = np.minimum(cos_ang, 1.0)
    cos_ang = np.maximum(cos_ang, -1.0)
    Verr_ang = np.arccos(cos_ang) * 180.0 / np.pi  # (1, Nt, Nz, Ny, Nx)
    # mask
    Vabs_GT_mask = Vabs_GT * V_mask
    Verr_abs = Verr_abs * V_mask
    Verr_ang = Verr_ang * V_mask
    # ------ get the maximum flow frame
    if frame_t is None:
        flow_time_arr = np.reshape(Vabs_GT_mask, (Nt, Nz * Ny * Nx))
        flow_time_arr = np.sum(flow_time_arr, axis=1, keepdims=False)  # (Nt,)
        flow_t = int(np.argmax(flow_time_arr))
    else:
        flow_t = frame_t
    # ------ only calculate error at the maximum flow frame
    denom = np.sum(V_mask)  # number of non-zero elements of the mask
    Vabs_GT_frame = Vabs_GT_mask[:, flow_t, :, :, :]
    Verr_abs_frame = Verr_abs[:, flow_t, :, :, :]
    Verr_ang_frame = Verr_ang[:, flow_t, :, :, :]
    # velocity abs
    RMSE_abs = np.sqrt(np.sum(Verr_abs_frame**2) / denom)
    # velocity angle
    meanErr_angle = np.sum(Verr_ang_frame) / denom
    return RMSE_abs, meanErr_angle