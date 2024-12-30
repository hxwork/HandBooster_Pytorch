from typing import Type, Any, Callable, Union, List, Optional, OrderedDict
import os
from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange
from model.mob_recon.models.modules import conv_layer, mobile_unit, linear_layer, Reorg
from model.mob_recon.models.resnet import FPN
from torchvision.models.resnet import ResNet
'''
Base on torch official ResNet architecture to reimplement a ResNetStack version backbone
    version: 1.11.0
'''


# copied from torchvision.models.resnet
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# end of copied


class ResNetStack1(nn.Module):

    def __init__(self, block, layers, norm_layer=None):
        super().__init__()
        # Params
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # dilate=replace_stride_with_dilation[0]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer4_up = self._make_layer_up(block, 512, 1024, layers[3], scale_factor=2)
        self.layer3_up = self._make_layer_up(block, 256, 512, layers[2], scale_factor=2)
        self.layer2_up = self._make_layer_up(block, 128, 256, layers[1], scale_factor=2)
        self.layer1_up = self._make_layer_up(block, 64, 64, layers[0])

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        return nn.Sequential(*layers)

    def _make_layer_up(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,  # BottleNect width
        outplanes: int,  # output channel
        blocks: int,  # how many sub-layers
        scale_factor: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        '''
        ResNet Stack 的 Upscale Layer
        scale_factor 是輸入 h, w 與輸出 h, w 的倍率
        '''
        # Params
        norm_layer = self._norm_layer  # default: nn.BatchNorm2d

        # Layers
        layers = []
        for _ in range(0, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),  # (#, 2048, 7, 7) -> (#, 2048, 14, 14)
                conv1x1(self.inplanes, outplanes),  # (#, 2048, 14, 14) -> (#, 1024, 14, 14)
                norm_layer(outplanes),
            ))

        self.inplanes = outplanes  # next layer

        return nn.Sequential(*layers)

    def forward(self, x):
        data = x
        layer1_out = self.layer1(x)  # (#, 64, 56, 56)
        layer2_out = self.layer2(layer1_out)  # (#, 512, 28, 28)
        layer3_out = self.layer3(layer2_out)  # (#, 1024, 14, 14)
        layer4_out = self.layer4(layer3_out)  # (#, 2048, 7, 7)

        x = layer3_out + self.layer4_up(layer4_out)
        x = layer2_out + self.layer3_up(x)
        x = layer1_out + self.layer2_up(x)
        x = data + self.layer1_up(x)

        return x


class ResNetStack2(nn.Module):

    def __init__(self, block, layers, norm_layer=None):
        super().__init__()
        # Params
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # dilate=replace_stride_with_dilation[0]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # (#, 2048, 7, 7)
        # self.inplanes=2048

        self.layer4_up = self._make_layer_up(block, 512, 1024, layers[3], scale_factor=2)
        # (#, 2048, 7, 7) -> (#, 512) -> (#, 512) -> (#, 2048) ...
        # (X) (#, 2048, 7, 7) -> (#, 256) -> (#, 256) -> (#, 1024)
        # upsample: (#, 2048, 7, 7) -> (#, 2048, 14, 14)

        # (#, 1024, 14, 14)
        self.layer3_up = self._make_layer_up(block, 256, 512, layers[2], scale_factor=2)
        # self.layer2_up = self._make_layer_up(block, 128, layers[1], scale_factor=2)
        # self.layer1_up = self._make_layer_up(block, 64, layers[0])

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        return nn.Sequential(*layers)

    def _make_layer_up(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,  # BottleNect width
        outplanes: int,  # output channel
        blocks: int,  # how many sub-layers
        scale_factor: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        '''
        ResNet Stack 的 Upscale Layer
        scale_factor 是輸入 h, w 與輸出 h, w 的倍率
        '''
        # Params
        downsample = None
        previous_dilation = 1
        norm_layer = self._norm_layer  # default: nn.BatchNorm2d

        # Layers
        # in layer 3.......................>
        # 512  -> (256, 256, 1024), stride=2, self.inplanes=1024
        # 1024 -> (256, 256, 1024), ...
        # up sample
        # 1024 -> (256, 256, 1024), ...
        # 1024 -> (256, 256, 512) + upsample=2, self.inplanes=64

        layers = []
        for _ in range(0, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))
        stride = 1

        # layers.append(
        #     block(
        #         self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
        #     )
        # )
        layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),  # (#, 2048, 7, 7) -> (#, 2048, 14, 14)
                conv1x1(self.inplanes, outplanes),  # (#, 2048, 14, 14) -> (#, 1024, 14, 14)
                norm_layer(outplanes),
            ))

        self.inplanes = outplanes  # next layer

        return nn.Sequential(*layers)

    def forward(self, x):
        # data = x
        layer1_out = self.layer1(x)  # (#, 64, 56, 56)
        layer2_out = self.layer2(layer1_out)  # (#, 512, 28, 28)
        layer3_out = self.layer3(layer2_out)  # (#, 1024, 14, 14)
        layer4_out = self.layer4(layer3_out)  # (#, 2048, 7, 7)

        x = layer3_out + self.layer4_up(layer4_out)
        x = layer2_out + self.layer3_up(x)

        return x, layer4_out


class ResnetStack_Backbone(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        kpts_num: int = 21,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        latent_size: int = 256,
        pretrain: bool = True,
    ) -> None:
        '''
        latent size: mapping value from latent output to Reg2DDecode3D
        '''
        super().__init__()
        # _log_api_usage_once(self)  commented this
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        # LAYERS
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet_stack1 = ResNetStack1(block, layers)
        self.resnet_stack2 = ResNetStack2(block, layers)

        stk2_out_channel = 512
        self.reduce = conv_layer(stk2_out_channel, kpts_num, 1, bn=False, relu=False)
        # self.avepool = nn.AvgPool2d(kernel_size=(2, 2))
        # reshape layer()
        self.uv_reg = nn.Sequential(linear_layer(256, 128, bn=False), linear_layer(128, 64, bn=False),
                                    linear_layer(64, 2, bn=False, relu=False))
        # mid branch
        self.mid_proj = conv_layer(2048, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain:
            # cur_dir = os.path.dirname(os.path.realpath(__file__))
            # weight = torch.load(os.path.join(cur_dir, '../out/resnetstack.pth'))
            # self.load_state_dict(weight, strict=False)
            self.update_weights()
            print('Load pre-trained weight: resnetstack.pth')

    def update_weights(self):
        '''
        pretrained: pretrained ResNet50 weights
        edited: edited model weights, model.state_dict()
        return: updated new model weights
        '''
        PATH_RESNET = '/uac/gds/xuhao/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
        print('torchvision - resnet50 weight found, loading...')
        pretrained_weights = torch.load(PATH_RESNET)

        # my model
        model = self.state_dict()

        # same_name = ['conv1', 'bn1']
        # stk1_name = ['layer1', 'layer2', 'layer3', 'layer4']
        # stk2_name = ['layer1', 'layer2', 'layer3', 'layer4']

        pretrained_layer_name_size = 60

        for k, v in pretrained_weights.items():
            names = k.split('.')

            if names[0] in ['conv1', 'bn1']:
                # same name
                assert (k in model)
                model[k] = v
                print(f'{k: <{pretrained_layer_name_size}}', end='')
                print(k)  # pretrained layer

            elif names[0] in ['layer1', 'layer2', 'layer3', 'layer4']:
                for i in range(1, 3):
                    new_layer_name = f'resnet_stack{i}.' + k
                    assert (new_layer_name in model)  # make sure the layer is in new model
                    model[new_layer_name] = v
                    print(f'{new_layer_name: <{pretrained_layer_name_size}}', end='')
                    print(k)  # pretrained layer

            else:  # no mapped layer
                print(' ' * pretrained_layer_name_size, end='')
                print(k)  # pretrained layer

        # torch.save(model, 'resnetstack.pth')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resnet_stack1(x)  # x: (#, 64, 56, 56)
        x, mid = self.resnet_stack2(x)  # x: (#, 512, 28, 28), mid: (#, 2048, 7, 7)
        x = self.reduce(x)  # x: (#, 21, 28, 28)
        # x = self.avepool(x)  # x: (#, 21, 14, 14)
        x = rearrange(x, 'b c h w -> b c (h w)')  # x: (#, 21, 196)
        uv_reg = self.uv_reg(x)

        latent = self.mid_proj(mid)
        return latent, uv_reg


class ResnetStack_Backbone_V1(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        kpts_num: int = 21,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        latent_size: int = 256,
        pretrain: bool = True,
    ) -> None:
        '''
        latent size: mapping value from latent output to Reg2DDecode3D
        '''
        super().__init__()
        # _log_api_usage_once(self)  commented this
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        # LAYERS
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet_stack1 = ResNetStack1(block, layers)
        self.resnet_stack2 = ResNetStack2(block, layers)

        stk2_out_channel = 512
        self.reduce = conv_layer(stk2_out_channel, kpts_num, 1, bn=False, relu=False)
        # self.avepool = nn.AvgPool2d(kernel_size=(2, 2))
        # reshape layer()
        self.uv_reg = nn.Sequential(linear_layer(256, 128, bn=False), linear_layer(128, 64, bn=False),
                                    linear_layer(64, 2, bn=False, relu=False))
        # mid branch
        self.mid_proj = conv_layer(2048, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)

        self.rot_pre_layer = FPN(pretrained=True)

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain:
            # cur_dir = os.path.dirname(os.path.realpath(__file__))
            # weight = torch.load(os.path.join(cur_dir, '../out/resnetstack.pth'))
            # self.load_state_dict(weight, strict=False)
            self.update_weights()
            print('Load pre-trained weight: resnetstack.pth')

    def update_weights(self):
        '''
        pretrained: pretrained ResNet50 weights
        edited: edited model weights, model.state_dict()
        return: updated new model weights
        '''
        PATH_RESNET = '/uac/gds/xuhao/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
        print('torchvision - resnet50 weight found, loading...')
        pretrained_weights = torch.load(PATH_RESNET)

        # my model
        model = self.state_dict()

        # same_name = ['conv1', 'bn1']
        # stk1_name = ['layer1', 'layer2', 'layer3', 'layer4']
        # stk2_name = ['layer1', 'layer2', 'layer3', 'layer4']

        pretrained_layer_name_size = 60

        for k, v in pretrained_weights.items():
            names = k.split('.')

            if names[0] in ['conv1', 'bn1']:
                # same name
                assert (k in model)
                model[k] = v
                print(f'{k: <{pretrained_layer_name_size}}', end='')
                print(k)  # pretrained layer

            elif names[0] in ['layer1', 'layer2', 'layer3', 'layer4']:
                for i in range(1, 3):
                    new_layer_name = f'resnet_stack{i}.' + k
                    assert (new_layer_name in model)  # make sure the layer is in new model
                    model[new_layer_name] = v
                    print(f'{new_layer_name: <{pretrained_layer_name_size}}', end='')
                    print(k)  # pretrained layer

            else:  # no mapped layer
                print(' ' * pretrained_layer_name_size, end='')
                print(k)  # pretrained layer

        # torch.save(model, 'resnetstack.pth')

    def forward(self, x):

        rot_pre_out = self.rot_pre_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resnet_stack1(x)  # x: (#, 64, 56, 56)
        x, mid = self.resnet_stack2(x)  # x: (#, 512, 28, 28), mid: (#, 2048, 7, 7)
        x = self.reduce(x)  # x: (#, 21, 28, 28)
        # x = self.avepool(x)  # x: (#, 21, 14, 14)
        x = rearrange(x, 'b c h w -> b c (h w)')  # x: (#, 21, 196)
        uv_reg = self.uv_reg(x)

        latent = self.mid_proj(mid)
        return latent, mid, rot_pre_out, uv_reg
