import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import cv2
import os
from torch.nn import functional as F
from scipy.linalg import orthogonal_procrustes
from hand_recon.common.tool import tensor_gpu
from hand_recon.common.utils.transforms import torch_pixel2cam
from hand_recon.common.utils.mano import MANO

right_mano = MANO(side='right')
left_mano = MANO(side='left')


class CoordLoss(nn.Module):

    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:, :, 2:] * is_3D[:, None, None].float()
            loss = torch.cat((loss[:, :, :2], loss_z), 2)

        return loss


class ParamLoss(nn.Module):

    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss


class NormalVectorLoss(nn.Module):

    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, is_valid=None):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        # valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))  # * valid_mask
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))  # * valid_mask
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))  # * valid_mask
        loss = torch.cat((cos1, cos2, cos3), 1)
        if is_valid is not None:
            loss *= is_valid
        return torch.mean(loss)


class EdgeLengthLoss(nn.Module):

    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, is_valid=None):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :])**2, 2, keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :])**2, 2, keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :])**2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :])**2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :])**2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :])**2, 2, keepdim=True))

        diff1 = torch.abs(d1_out - d1_gt)  # * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt)  # * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt)  # * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3), 1)
        if is_valid is not None:
            loss *= is_valid
        return torch.mean(loss)


def align_w_scale(mtx1, mtx2, return_trafo=False):
    ''' Align the predicted entity in some optimality sense with the ground truth. '''
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def torch_align_w_scale(mtx1, mtx2, return_trafo=False):
    ''' Align the predicted entity in some optimality sense with the ground truth. '''
    # center
    t1 = torch.mean(mtx1, dim=1, keepdim=True)
    t2 = torch.mean(mtx2, dim=1, keepdim=True)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = torch.linalg.matrix_norm(mtx1_t, dim=(-2, -1), keepdim=True) + 1e-8
    mtx1_t /= s1
    s2 = torch.linalg.matrix_norm(mtx2_t, dim=(-2, -1), keepdim=True) + 1e-8
    mtx2_t /= s2

    # orth alignment
    u, w, v = torch.svd(torch.matmul(mtx2_t.transpose(1, 2), mtx1_t).transpose(1, 2))
    R = torch.matmul(u, v.transpose(1, 2))
    s = torch.sum(w, dim=1, keepdim=True).unsqueeze(-1)
    # apply trafos to the second matrix
    mtx2_t = torch.matmul(mtx2_t, R.transpose(1, 2)) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def filter_data_by_metric(input, output, save_img=False):
    metric = {}

    verts_out = output['pred_verts3d_cam']
    joints_out = output['pred_joints3d_cam']
    verts_out = verts_out - joints_out[:, 0, None]
    joints_out = joints_out - joints_out[:, 0, None]

    # GT and rigid align
    verts_gt = output['gt_verts3d_cam'].repeat(2, 1, 1)
    joints_gt = output['gt_joints3d_cam'].repeat(2, 1, 1)
    verts_gt = verts_gt - joints_gt[:, 0, None]
    joints_gt = joints_gt - joints_gt[:, 0, None]

    # align predictions
    joints_out_aligned = torch_align_w_scale(joints_gt, joints_out)
    verts_out_aligned = torch_align_w_scale(verts_gt, verts_out)

    # # m to mm
    MPJPE = torch.mean(torch.sqrt(torch.sum((joints_out - joints_gt)**2, dim=2)), dim=1) * 1000
    PA_MPJPE = torch.mean(torch.sqrt(torch.sum((joints_out_aligned - joints_gt)**2, dim=2)), dim=1) * 1000
    MPVPE = torch.mean(torch.sqrt(torch.sum((verts_out - verts_gt)**2, dim=2)), dim=1) * 1000
    PA_MPVPE = torch.mean(torch.sqrt(torch.sum((verts_out_aligned - verts_gt)**2, dim=2)), dim=1) * 1000
    score = MPJPE + PA_MPJPE + MPVPE + PA_MPVPE

    ori_MPJPE, gen_MPJPE = torch.chunk(MPJPE, 2, dim=0)
    ori_PA_MPJPE, gen_PA_MPJPE = torch.chunk(PA_MPJPE, 2, dim=0)
    ori_MPVPE, gen_MPVPE = torch.chunk(MPVPE, 2, dim=0)
    ori_PA_MPVPE, gen_PA_MPVPE = torch.chunk(PA_MPVPE, 2, dim=0)
    ori_score, gen_score = torch.chunk(score, 2, dim=0)

    metric = {
        'ori_MPJPE': ori_MPJPE,
        'gen_MPJPE': gen_MPJPE,
        'ori_PA_MPJPE': ori_PA_MPJPE,
        'gen_PA_MPJPE': gen_PA_MPJPE,
        'ori_MPVPE': ori_MPVPE,
        'gen_MPVPE': gen_MPVPE,
        'ori_PA_MPVPE': ori_PA_MPVPE,
        'gen_PA_MPVPE': gen_PA_MPVPE,
        'ori_score': ori_score,
        'gen_score': gen_score,
    }

    verts_ori, verts_gen = torch.chunk(verts_out, 2, dim=0)
    verts_gt = torch.chunk(verts_gt, 2, dim=0)[0]
    verts = {
        'verts_ori': verts_ori,
        'verts_gen': verts_gen,
        'verts_gt': verts_gt,
    }

    return metric, verts
