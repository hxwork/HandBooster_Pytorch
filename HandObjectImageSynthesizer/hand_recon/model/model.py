import logging
import torch
import os
import time
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from contextlib import nullcontext
# from pytorch3d import transforms as p3dt
from torchvision import ops

from hand_recon.common.utils.transforms import torch_cam2pixel
from hand_recon.common.utils.mano import MANO
from hand_recon.model.hand_occ_net import backbone, transformer, regressor
from hand_recon.model.hand_occ_net.mano_head import batch_rodrigues

from hand_recon.model.mob_recon.utils.read import spiral_tramsform
from hand_recon.model.mob_recon.conv.dsconv import DSConv
from hand_recon.model.mob_recon.conv.spiralconv import SpiralConv
from hand_recon.model.mob_recon.models.densestack import DenseStack_Backnone
from hand_recon.model.mob_recon.models.modules import Reg2DDecode3D

logger = logging.getLogger(__name__)
mano = MANO()


# Init model weights
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


# Define the model of some methods
#####################################################################################


class MobRecon_DS(nn.Module):

    def __init__(self):
        super(MobRecon_DS, self).__init__()
        latent_size = 256
        self.backbone = DenseStack_Backnone(latent_size=latent_size, kpts_num=21, pretrain=True)
        template_fp = './hand_recon/model/mob_recon/template/template.ply'
        transform_fp = './hand_recon/model/mob_recon/template/transform.pkl'
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp, template_fp, [2, 2, 2, 2], [9, 9, 9, 9], [1, 1, 1, 1])
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        self.decoder3d = Reg2DDecode3D(latent_size=latent_size,
                                       out_channels=[32, 64, 128, 256],
                                       spiral_indices=spiral_indices,
                                       up_transform=up_transform,
                                       uv_channel=21,
                                       meshconv=(SpiralConv, DSConv)[False])
        self.mano_pose_size = 16 * 3
        self.mano_layer = mano.layer
        self.mano_joint_reg = torch.from_numpy(mano.joint_regressor)

    def forward(self, input):
        if 'ori_img' in input:
            x = torch.cat((input['ori_img'], input['img']), dim=0)
        else:
            x = input['img']

        latent, pred2d_pt = self.backbone(x)
        pred3d = self.decoder3d(pred2d_pt, latent)

        if 'mano_pose' in input:
            gt_mano_params = torch.cat([input['mano_pose'], input['mano_shape']], dim=1)
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000
        else:
            gt_mano_params = None

        output = {}
        output['pred_verts3d_cam'] = pred3d
        output['pred_joints3d_cam'] = torch.matmul(self.mano_joint_reg.to(pred3d.device), pred3d)
        output['pred_joints_img'] = pred2d_pt

        # # for eval
        # output['pred_joints3d_cam'] = torch.matmul(self.mano_joint_reg.to(pred3d.device), pred3d)
        # output['pred_verts3d_cam'] = pred3d

        if gt_mano_params is not None:
            output['gt_verts3d_cam'] = gt_verts

            if 'val_mano_pose' in input:
                # for eval
                val_gt_verts, val_gt_joints = self.mano_layer(th_pose_coeffs=input['val_mano_pose'], th_betas=input['mano_shape'])
                output['gt_verts3d_cam'], output['gt_joints3d_cam'] = val_gt_verts / 1000, val_gt_joints / 1000
        return output


def fetch_model(model_name, model_path):
    if model_name == 'mobrecon':
        model = MobRecon_DS()

    else:
        raise NotImplementedError

    state = torch.load(model_path)

    try:
        model.load_state_dict(state['state_dict'])
    except RuntimeError:
        print('Using custom loading net')
        net_dict = model.state_dict()
        if 'module' in list(state['state_dict'].keys())[0]:
            state_dict = {k[7:]: v for k, v in state['state_dict'].items() if k not in net_dict.keys()}
        net_dict.update(state_dict)
        model.load_state_dict(net_dict, strict=False)

    return model
