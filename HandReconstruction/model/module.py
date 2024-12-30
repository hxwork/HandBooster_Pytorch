import logging
import math
from signal import alarm
from sqlite3 import paramstyle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import numpy as np
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
from collections import OrderedDict
from model.layer import *

logger = logging.getLogger(__name__)


class ResNetBackbone(nn.Module):

    def __init__(self, cfg):
        self.cfg = cfg
        self.resnet_type = self.cfg.model.resnet_type
        resnet_spec = {
            18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
            34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
            50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
            101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
            152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], "resnet152")
        }
        block, layers, channels, name = resnet_spec[self.resnet_type]

        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                # nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, skip_early=False):
        if not skip_early:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x, x4

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add "None" in case of no fc layer, that will raise error
        org_resnet.pop("fc.weight", None)
        org_resnet.pop("fc.bias", None)

        self.load_state_dict(org_resnet)
        logger.info("Initialize resnet from model zoo")


class ResNetBackboneNoBN(nn.Module):

    def __init__(self, cfg):
        self.cfg = cfg
        self.resnet_type = self.cfg.model.resnet_type
        resnet_spec = {
            18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
            34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
            50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
            101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
            152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], "resnet152")
        }
        block, layers, channels, name = resnet_spec[self.resnet_type]

        self.name = name
        self.inplanes = 64
        super(ResNetBackboneNoBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                # nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, skip_early=False):
        if not skip_early:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x, x4

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add "None" in case of no fc layer, that will raise error
        org_resnet.pop("fc.weight", None)
        org_resnet.pop("fc.bias", None)

        self.load_state_dict(org_resnet)
        logger.info("Initialize resnet from model zoo")


class PoseNet(nn.Module):

    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        self.cfg = cfg
        self.joint_num = self.cfg.model.joint_num
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.conv_x = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256 * self.cfg.data.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)

        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)

        # z axis
        img_feat_z = img_feat.mean((2, 3))[:, :, None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1, 256, self.cfg.data.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z), 2)
        return joint_coord


class Pose2Feat(nn.Module):

    def __init__(self, cfg):
        super(Pose2Feat, self).__init__()
        self.cfg = cfg
        self.joint_num = self.cfg.model.joint_num
        self.conv = make_conv_layers([64 + self.joint_num * cfg.data.output_hm_shape[0], 64])

    def forward(self, img_feat, joint_heatmap_3d):
        joint_heatmap_3d = joint_heatmap_3d.view(-1, self.joint_num * self.cfg.data.output_hm_shape[0], self.cfg.data.output_hm_shape[1], self.cfg.data.output_hm_shape[2])
        feat = torch.cat((img_feat, joint_heatmap_3d), 1)
        feat = self.conv(feat)
        return feat


class MeshNet(nn.Module):

    def __init__(self, cfg):
        super(MeshNet, self).__init__()
        self.cfg = cfg
        self.vertex_num = self.cfg.model.vertex_num
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.conv_x = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256 * self.cfg.data.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)

        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)

        # z axis
        img_feat_z = img_feat.mean((2, 3))[:, :, None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1, 256, self.cfg.data.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        mesh_coord = torch.cat((coord_x, coord_y, coord_z), 2)
        return mesh_coord


class ParamRegressor(nn.Module):

    def __init__(self, cfg):
        super(ParamRegressor, self).__init__()
        self.cfg = cfg
        self.joint_num = self.cfg.model.joint_num
        self.fc = make_linear_layers([self.joint_num * 3, 1024, 512], use_bn=True)
        self.fc_pose = make_linear_layers([512, 16 * 6], relu_final=False)  # hand joint orientation
        self.fc_shape = make_linear_layers([512, 10], relu_final=False)  # shape parameter

    def rot6d_to_rotmat(self, x):
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose_3d):
        batch_size = pose_3d.shape[0]
        pose_3d = pose_3d.view(-1, self.joint_num * 3)
        feat = self.fc(pose_3d)

        pose = self.fc_pose(feat)
        pose = self.rot6d_to_rotmat(pose)
        pose = torch.cat([pose, torch.zeros((pose.shape[0], 3, 1)).cuda().float()], 2)
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch_size, -1)

        shape = self.fc_shape(feat)

        return pose, shape


class ParamLightPred(nn.Module):

    def __init__(self, cfg):
        super(ParamLightPred, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetBackbone(self.cfg)
        # self.fc = make_linear_layers([2048, 3072, 512], use_bn=True)

        self.l_reg = make_linear_layers([2048, 1024, 512, self.cfg.model.num_lights * 3], use_bn=True, relu_final=False)
        self.s_reg = make_linear_layers([2048, 1024, 512, self.cfg.model.num_lights], use_bn=True, relu_final=False)
        self.c_reg = make_linear_layers([2048, 1024, 512, self.cfg.model.num_lights * 3], use_bn=True, relu_final=False)
        self.a_reg = make_linear_layers([2048, 1024, 512, 3], use_bn=True, relu_final=False)
        # self.init_weights()

    def init_weights(self):
        pass
        # self.backbone.init_weights()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias.data, 0)
        #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight.data, 1)
        #         nn.init.constant_(m.bias.data, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        #         nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, decode_vec = self.backbone(x)
        decode_vec = F.adaptive_avg_pool2d(decode_vec, (1, 1))  # output: 1*1, average pooling over full feature map
        decode_vec = torch.flatten(decode_vec, 1)  # flatten: keep some dims and merge the others
        # decode_vec = self.fc(decode_vec)

        param_l = self.l_reg(decode_vec)  # (B, 3*N)
        param_s = self.s_reg(decode_vec)  # (B, N)
        param_c = self.c_reg(decode_vec)  # (B, 3*N)
        ambient = self.a_reg(decode_vec)  # (B, 3), RGB value

        param_l = param_l.view(-1, 3, self.cfg.model.num_lights)  # (B, 3, N)
        param_c = param_c.view(-1, 3, self.cfg.model.num_lights)  # (B, 3, N)

        param_l = F.normalize(param_l, p=2, dim=1)  # convert to unit vector
        param_s = torch.sigmoid(param_s) * 5
        param_c = torch.sigmoid(param_c) + 1e-5
        ambient = torch.sigmoid(ambient) + 1e-5

        sg_coeffs = torch.cat((param_l, param_s.unsqueeze(1), param_c), dim=1)  # (B, 7, N)
        return sg_coeffs, ambient


class ParamLightPredWeights(nn.Module):

    def __init__(self, cfg):
        super(ParamLightPredWeights, self).__init__()
        self.cfg = cfg
        self.backbone = ResNetBackbone(self.cfg)
        # self.fc = make_linear_layers([2048, 3072, 512], use_bn=True)
        self.lobe = torch.load("/research/d4/rshr/xuhao/code/ARHandLightDP/common/utils/sphere_locations/sphere_locations_N{}.pth".format(self.cfg.model.num_lights)).cuda().float()
        # self.l_reg = make_linear_layers([2048, 1024, 512, self.cfg.model.num_lights], use_bn=True, relu_final=False)
        self.s_reg = make_linear_layers([2048, 1024, self.cfg.model.num_lights], use_bn=True, relu_final=False)
        self.r_branch = make_linear_layers([2048, 1024, self.cfg.model.num_lights], use_bn=True, relu_final=False)
        self.g_branch = make_linear_layers([2048, 1024, self.cfg.model.num_lights], use_bn=True, relu_final=False)
        self.b_branch = make_linear_layers([2048, 1024, self.cfg.model.num_lights], use_bn=True, relu_final=False)
        # self.a_reg = make_linear_layers([2048, 1024, 3], use_bn=True, relu_final=False)
        # self.init_weights()

    def init_weights(self):
        # pass
        # self.backbone.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, decode_vec = self.backbone(x)
        decode_vec = F.adaptive_avg_pool2d(decode_vec, (1, 1))  # output: 1*1, average pooling over full feature map
        decode_vec = torch.flatten(decode_vec, 1)  # flatten: keep some dims and merge the others
        # decode_vec = self.fc(decode_vec)

        # param_l = self.l_reg(decode_vec)  # (B, 3*N)
        s_weights = self.s_reg(decode_vec)  # (B, N)
        # c_weights = self.c_reg(decode_vec)  # (B, 3*N)
        r_weights = self.r_branch(decode_vec)  # (B, N)
        g_weights = self.g_branch(decode_vec)  # (B, N)
        b_weights = self.b_branch(decode_vec)  # (B, N)
        c_weights = torch.stack((r_weights, g_weights, b_weights), dim=-1)  # (B, N, 3)
        # ambient = self.a_reg(decode_vec)  # (B, 3), RGB value
        ambient = None  # dummy

        # param_l = param_l.view(-1, 3, self.cfg.model.num_lights)  # (B, 3, N)
        # c_weights = c_weights.view(-1, self.cfg.model.num_lights, 3)  # (B, N, 3)

        # param_l = F.normalize(param_l, p=2, dim=1)  # convert to unit vector
        s_weights = torch.abs(s_weights)
        c_weights = torch.abs(c_weights)
        # l_weights = torch.ones_like(c_weights, requires_grad=False).to(c_weights.device)
        # ambient = torch.sigmoid(ambient) + 1e-5

        sg_coeffs_weights = torch.cat((self.lobe.repeat(s_weights.size()[0], 1, 1), s_weights.unsqueeze(-1), c_weights), dim=2)  # (B, N, 7)
        return sg_coeffs_weights, ambient


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),

        # If the bottle neck mode is set, apply feature reduction to limit the growth of features
        # Why should we expand the number of features by bn_size*growth?

        # https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        if bn_size > 0:
            interChannels = 4 * growth_rate
            self.add_module('conv1', nn.Conv2d(num_input_features, interChannels, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(interChannels))
            self.add_module('conv2', nn.Conv2d(interChannels, growth_rate, kernel_size=3, padding=1, bias=False))
        else:
            self.add_module('conv2', nn.Conv2d(num_input_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class OriDenseNet(nn.Module):
    """Densenet-BC model class, based on
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        growth_rate=12, block_config=(16, 16, 16), compression=0.5,num_init_features=24
        , bn_size=4, drop_rate=0, avgpool_size=8,
    """

    def __init__(self, cfg, growth_rate=12, block_config=(16, 16, 16), compression=0.5, num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=4):
        super(OriDenseNet, self).__init__()
        self.cfg = cfg

        self.avgpool_size = avgpool_size
        # The first Convolution layer
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
            ]))
        # Did not add the pooling layer to preserve dimension
        # The number of layers in each Densnet is adjustable

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            Dense_block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # Add name to the Denseblock
            self.features.add_module('denseblock%d' % (i + 1), Dense_block)

            # Increase the number of features by the growth rate times the number
            # of layers in each Denseblock
            num_features += num_layers * growth_rate

            # check whether the current block is the last block
            # Add a transition layer to all Denseblocks except the last
            if i != len(block_config):
                # Reduce the number of output features in the transition layer

                nOutChannels = int(math.floor(num_features * compression))

                trans = _Transition(num_input_features=num_features, num_output_features=nOutChannels)
                self.features.add_module('transition%d' % (i + 1), trans)
                # change the number of features for the next Dense block
                num_features = nOutChannels

            # Final batch norm
            self.features.add_module('last_norm%d' % (i + 1), nn.BatchNorm2d(num_features))

        # Linear layer
        self.fc = nn.Linear(8208, 1024)
        self.dist_reg = nn.Linear(1024, self.cfg.model.num_lights)
        self.rgb_ratio_reg = nn.Linear(1024, 3)
        self.intensity_reg = nn.Linear(1024, 1)
        self.ambient_reg = nn.Linear(1024, 3)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def init_weights(self):
        # pass
        # self.backbone.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.fc(out)

        distribution = self.dist_reg(out)  # (B, N)
        rgb_ratio = self.rgb_ratio_reg(out)  # (B, 3)
        intensity = self.intensity_reg(out)  # (B, 1)
        ambient = self.ambient_reg(out)  # (B, 3)

        # distribution = torch.abs(distribution)
        # # distribution = distribution / torch.norm(distribution, dim=1, keepdim=True)
        # rgb_ratio = torch.abs(rgb_ratio)
        # intensity = torch.abs(intensity)
        # ambient = torch.abs(ambient)

        return distribution, rgb_ratio, intensity, ambient


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        growth_rate=12, block_config=(16, 16, 16), compression=0.5,num_init_features=24
        , bn_size=4, drop_rate=0, avgpool_size=8,
    """

    def __init__(self, cfg, growth_rate=12, block_config=(16, 16, 16), compression=0.5, num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=5):
        super(DenseNet, self).__init__()
        self.cfg = cfg

        self.avgpool_size = avgpool_size
        # The first Convolution layer
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
            ]))
        # Did not add the pooling layer to preserve dimension
        # The number of layers in each Densnet is adjustable

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            Dense_block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # Add name to the Denseblock
            self.features.add_module('denseblock%d' % (i + 1), Dense_block)

            # Increase the number of features by the growth rate times the number
            # of layers in each Denseblock
            num_features += num_layers * growth_rate

            # check whether the current block is the last block
            # Add a transition layer to all Denseblocks except the last
            if i != len(block_config):
                # Reduce the number of output features in the transition layer

                nOutChannels = int(math.floor(num_features * compression))

                trans = _Transition(num_input_features=num_features, num_output_features=nOutChannels)
                self.features.add_module('transition%d' % (i + 1), trans)
                # change the number of features for the next Dense block
                num_features = nOutChannels

            # Final batch norm
            self.features.add_module('last_norm%d' % (i + 1), nn.BatchNorm2d(num_features))

        # Linear layer
        self.fc = nn.Linear(6156, 1024)
        self.dist_reg = nn.Linear(1024, self.cfg.model.num_lights)
        self.rgb_ratio_reg = nn.Linear(1024, 3)
        self.intensity_reg = nn.Linear(1024, 1)
        self.ambient_reg = nn.Linear(1024, 3)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def init_weights(self):
        # pass
        # self.backbone.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.fc(out)

        distribution = self.dist_reg(out)  # (B, N)
        rgb_ratio = self.rgb_ratio_reg(out)  # (B, 3)
        intensity = self.intensity_reg(out)  # (B, 1)
        ambient = self.ambient_reg(out)  # (B, 3)

        # distribution = torch.abs(distribution)
        # # distribution = distribution / torch.norm(distribution, dim=1, keepdim=True)
        # rgb_ratio = torch.abs(rgb_ratio)
        # intensity = torch.abs(intensity)
        # ambient = torch.abs(ambient)

        return distribution, rgb_ratio, intensity, ambient
