import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import cv2
import os
from torch.nn import functional as F
from scipy.linalg import orthogonal_procrustes
from common.tool import tensor_gpu
from common.utils.transforms import torch_pixel2cam
from common.utils.mano import MANO
from common.utils.vis import render_badcase_mesh, render_filter_mesh

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


def compute_loss(cfg, input, output):

    loss = {}
    if cfg.model.name == 'hand_occ_net':
        loss['mano_verts'] = 1e4 * F.mse_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam'])
        loss['mano_joints'] = 1e4 * F.mse_loss(output['pred_joints3d_cam'], output['gt_joints3d_cam'])
        loss['mano_pose'] = 10 * F.mse_loss(output['pred_mano_pose'], output['gt_mano_pose'])
        loss['mano_shape'] = 0.1 * F.mse_loss(output['pred_mano_shape'], output['gt_mano_shape'])
        loss['joints_img'] = 100 * F.mse_loss(output['pred_joints_img'], input['joints_img'])

    elif cfg.model.name == 'mobrecon':
        normal_loss = NormalVectorLoss(right_mano.face)
        edge_loss = EdgeLengthLoss(right_mano.face)
        loss['verts_loss'] = F.l1_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam'])
        loss['joint_img_loss'] = F.l1_loss(output['pred_joints_img'], input['joints_img'])
        loss['normal_loss'] = 0.1 * normal_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam'])
        loss['edge_loss'] = edge_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam'])

    elif cfg.model.name == 'h2onet':
        normal_loss = NormalVectorLoss(right_mano.face)
        edge_loss = EdgeLengthLoss(right_mano.face)
        loss['verts_wo_gr_loss'] = F.l1_loss(output['pred_verts3d_wo_gr'], output['gt_verts3d_wo_gr'])
        loss['joints_wo_gr_loss'] = F.l1_loss(output['pred_joints3d_wo_gr'], output['gt_joints3d_wo_gr'])
        loss['normal_wo_gr_loss'] = 0.1 * normal_loss(output['pred_verts3d_wo_gr'], output['gt_verts3d_wo_gr'])
        loss['edge_wo_gr_loss'] = edge_loss(output['pred_verts3d_wo_gr'], output['gt_verts3d_wo_gr'])
        loss['joint_img_loss'] = F.l1_loss(output['pred_joints_img'], input['joints_img'])
        if cfg.loss.w_gr:
            loss['verts_w_gr_loss'] = F.l1_loss(output['pred_verts3d_w_gr'], output['gt_verts3d_w_gr'])
            loss['joints_w_gr_loss'] = F.l1_loss(output['pred_joints3d_w_gr'], output['gt_joints3d_w_gr'])
            loss['glob_rot_loss'] = F.mse_loss(output['pred_glob_rot_mat'], output['gt_glob_rot_mat'])

    else:
        raise NotImplementedError

    # Common operation
    loss = {k: loss[k].mean() for k in loss}
    loss['total'] = sum(loss[k] for k in loss)
    return loss


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


def compute_occ_acc(pred_occ, gt_occ):
    pred_occ = torch.argmax(pred_occ, dim=1)  # (6B, 1)
    pred_occ = torch.chunk(pred_occ, 6, dim=0)  # 6 * (B, 1)
    gt_occ = torch.chunk(gt_occ, 6, dim=0)  # 6 * (B, 1)
    occ_acc = {}
    finger_names = ['thumb', 'index', 'middle', 'ring', 'little', 'palm']
    for pred, gt, name in zip(pred_occ, gt_occ, finger_names):
        occ_acc[name] = torch.mean((pred == gt).float())
    return occ_acc


def compute_metric(cfg, input, output):
    metric = {}
    if cfg.model.name == 'hand_occ_net':
        metric['mano_verts'] = cfg.loss.lambda_mano_verts * F.mse_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam'])
        metric['mano_joints'] = cfg.loss.lambda_mano_joints * F.mse_loss(output['pred_joints3d_cam'], output['gt_joints3d_cam'])
        metric['mano_pose'] = cfg.loss.lambda_mano_pose * F.mse_loss(output['pred_mano_pose'], output['gt_mano_pose'])
        metric['mano_shape'] = cfg.loss.lambda_mano_shape * F.mse_loss(output['pred_mano_shape'], output['gt_mano_shape'])
        metric['joints_img'] = cfg.loss.lambda_joints_img * F.mse_loss(output['pred_joints_img'], input['joints_img'])

        verts_out = output['pred_verts3d_cam']
        joints_out = output['pred_joints3d_cam']
        verts_out = verts_out - joints_out[:, 0, None]
        joints_out = joints_out - joints_out[:, 0, None]

        # # GT and rigid align
        verts_gt = output['gt_verts3d_cam']
        joints_gt = output['gt_joints3d_cam']
        verts_gt = verts_gt - joints_gt[:, 0, None]
        joints_gt = joints_gt - joints_gt[:, 0, None]

        # align predictions
        joints_out_aligned = torch_align_w_scale(joints_gt, joints_out)
        verts_out_aligned = torch_align_w_scale(verts_gt, verts_out)

        # m to mm
        joints_out *= 1000
        joints_out_aligned *= 1000
        joints_gt *= 1000
        verts_out *= 1000
        verts_out_aligned *= 1000
        verts_gt *= 1000

        metric['MPJPE'] = torch.sqrt(torch.sum((joints_out - joints_gt)**2, 2)).mean()
        metric['PA-MPJPE'] = torch.sqrt(torch.sum((joints_out_aligned - joints_gt)**2, 2)).mean()
        metric['MPVPE'] = torch.sqrt(torch.sum((verts_out - verts_gt)**2, 2)).mean()
        metric['PA-MPVPE'] = torch.sqrt(torch.sum((verts_out_aligned - verts_gt)**2, 2)).mean()
        metric['score'] = metric['MPJPE'] + metric['PA-MPJPE'] + metric['MPVPE'] + metric['PA-MPVPE']

    elif cfg.model.name == 'mobrecon':
        normal_loss = NormalVectorLoss(right_mano.face)
        edge_loss = EdgeLengthLoss(right_mano.face)
        metric['verts_loss'] = F.l1_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam']).mean()
        metric['joint_img_loss'] = F.l1_loss(output['pred_joints_img'], input['joints_img']).mean()
        metric['normal_loss'] = 0.1 * normal_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam']).mean()
        metric['edge_loss'] = edge_loss(output['pred_verts3d_cam'], output['gt_verts3d_cam']).mean()

        verts_out = output['pred_verts3d_cam']
        joints_out = output['pred_joints3d_cam']
        verts_out = verts_out - joints_out[:, 0, None]
        joints_out = joints_out - joints_out[:, 0, None]

        # # GT and rigid align
        verts_gt = output['gt_verts3d_cam']
        joints_gt = output['gt_joints3d_cam']
        verts_gt = verts_gt - joints_gt[:, 0, None]
        joints_gt = joints_gt - joints_gt[:, 0, None]

        # align predictions
        joints_out_aligned = torch_align_w_scale(joints_gt, joints_out)
        verts_out_aligned = torch_align_w_scale(verts_gt, verts_out)

        # m to mm
        joints_out *= 1000
        joints_out_aligned *= 1000
        joints_gt *= 1000
        verts_out *= 1000
        verts_out_aligned *= 1000
        verts_gt *= 1000

        metric['MPJPE'] = torch.sqrt(torch.sum((joints_out - joints_gt)**2, 2)).mean()
        metric['PA-MPJPE'] = torch.sqrt(torch.sum((joints_out_aligned - joints_gt)**2, 2)).mean()
        metric['MPVPE'] = torch.sqrt(torch.sum((verts_out - verts_gt)**2, 2)).mean()
        metric['PA-MPVPE'] = torch.sqrt(torch.sum((verts_out_aligned - verts_gt)**2, 2)).mean()
        metric['score'] = metric['MPJPE'] + metric['PA-MPJPE'] + metric['MPVPE'] + metric['PA-MPVPE']

    elif cfg.model.name == 'h2onet':
        normal_loss = NormalVectorLoss(right_mano.face)
        edge_loss = EdgeLengthLoss(right_mano.face)
        metric['verts_wo_gr'] = F.l1_loss(output['pred_verts3d_wo_gr'], output['gt_verts3d_wo_gr'])
        metric['verts_w_gr'] = F.l1_loss(output['pred_verts3d_w_gr'], output['gt_verts3d_w_gr'])
        metric['joints_wo_gr'] = F.l1_loss(output['pred_joints3d_wo_gr'], output['gt_joints3d_wo_gr'])
        metric['joints_w_gr'] = F.l1_loss(output['pred_joints3d_w_gr'], output['gt_joints3d_w_gr'])
        metric['glob_rot_loss'] = F.mse_loss(output['pred_glob_rot_mat'], output['gt_glob_rot_mat'])
        metric['joint_img'] = F.l1_loss(output['pred_joints_img'], input['joints_img'])
        metric['normal'] = 0.1 * normal_loss(output['pred_verts3d_wo_gr'], output['gt_verts3d_wo_gr'])
        metric['edge'] = edge_loss(output['pred_verts3d_wo_gr'], output['gt_verts3d_wo_gr'])

        verts_out = output['pred_verts3d_w_gr']
        joints_out = output['pred_joints3d_w_gr']
        verts_out = verts_out - joints_out[:, 0, None]
        joints_out = joints_out - joints_out[:, 0, None]

        # # GT and rigid align
        verts_gt = output['gt_verts3d_w_gr']
        joints_gt = output['gt_joints3d_w_gr']
        verts_gt = verts_gt - joints_gt[:, 0, None]
        joints_gt = joints_gt - joints_gt[:, 0, None]

        # align predictions
        joints_out_aligned = torch_align_w_scale(joints_gt, joints_out)
        verts_out_aligned = torch_align_w_scale(verts_gt, verts_out)

        # m to mm
        joints_out *= 1000
        joints_out_aligned *= 1000
        joints_gt *= 1000
        verts_out *= 1000
        verts_out_aligned *= 1000
        verts_gt *= 1000
        metric['MPJPE'] = torch.sqrt(torch.sum((joints_out - joints_gt)**2, 2)).mean()
        metric['PA-MPJPE'] = torch.sqrt(torch.sum((joints_out_aligned - joints_gt)**2, 2)).mean()
        metric['MPVPE'] = torch.sqrt(torch.sum((verts_out - verts_gt)**2, 2)).mean()
        metric['PA-MPVPE'] = torch.sqrt(torch.sum((verts_out_aligned - verts_gt)**2, 2)).mean()
        if cfg.loss.w_gr:
            metric['score'] = metric['MPJPE'] + metric['PA-MPJPE'] + metric['MPVPE'] + metric['PA-MPVPE']
        else:
            metric['score'] = metric['PA-MPJPE'] + metric['PA-MPVPE']

    else:
        raise NotImplementedError

    return metric
