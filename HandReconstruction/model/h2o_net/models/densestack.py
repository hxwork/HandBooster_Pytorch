import torch
import torch.nn as nn
import logging
from model.h2o_net.models.modules import conv_layer, mobile_unit, linear_layer, Reorg

logger = logging.getLogger(__name__)


class DenseBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in // 4)
        self.conv2 = mobile_unit(channel_in * 5 // 4, channel_in // 4)
        self.conv3 = mobile_unit(channel_in * 6 // 4, channel_in // 4)
        self.conv4 = mobile_unit(channel_in * 7 // 4, channel_in // 4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3), dim=1)
        out4 = self.conv4(comb3)
        comb4 = torch.cat((comb3, out4), dim=1)
        return comb4


class DenseBlock2(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in // 2)
        self.conv2 = mobile_unit(channel_in * 3 // 2, channel_in // 2)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), dim=1)
        return comb2


class DenseBlock3(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock3, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in)
        self.conv2 = mobile_unit(channel_in * 2, channel_in)
        self.conv3 = mobile_unit(channel_in * 3, channel_in)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((comb1, out2), dim=1)
        out3 = self.conv3(comb2)
        comb3 = torch.cat((comb2, out3), dim=1)
        return comb3


class DenseBlock2_noExpand(nn.Module):
    dump_patches = True

    def __init__(self, channel_in):
        super(DenseBlock2_noExpand, self).__init__()
        self.channel_in = channel_in
        self.conv1 = mobile_unit(channel_in, channel_in * 3 // 4)
        self.conv2 = mobile_unit(channel_in * 7 // 4, channel_in // 4)

    def forward(self, x):
        out1 = self.conv1(x)
        comb1 = torch.cat((x, out1), dim=1)
        out2 = self.conv2(comb1)
        comb2 = torch.cat((out1, out2), dim=1)
        return comb2


class SenetBlock(nn.Module):
    dump_patches = True

    def __init__(self, channel, size):
        super(SenetBlock, self).__init__()
        self.size = size
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel = channel
        self.fc1 = linear_layer(self.channel, min(self.channel // 2, 256))
        self.fc2 = linear_layer(min(self.channel // 2, 256), self.channel, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_out = x
        pool = self.globalAvgPool(x)
        pool = pool.view(pool.size(0), -1)
        fc1 = self.fc1(pool)
        out = self.fc2(fc1)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out * original_out


class DenseStack(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel):
        super(DenseStack, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel * 4)
        self.senet3 = SenetBlock(input_channel * 8, 8)
        self.transition3 = nn.AvgPool2d(2)
        # self.dense4 = DenseBlock2_noExpand(input_channel * 8)
        # self.dense5 = DenseBlock2_noExpand(input_channel * 8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel * 8, input_channel * 4, num3x3=1), mobile_unit(input_channel * 4, input_channel * 4, num3x3=2))
        self.senet4 = SenetBlock(input_channel * 4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink2 = nn.Sequential(mobile_unit(input_channel * 4, input_channel * 2, num3x3=1), mobile_unit(input_channel * 2, input_channel * 2, num3x3=2))
        self.senet5 = SenetBlock(input_channel * 2, 8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink3 = nn.Sequential(mobile_unit(input_channel * 2, input_channel * 2, num3x3=1), mobile_unit(input_channel * 2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel, 16)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.dense3(d2))
        u1 = self.upsample1(self.senet4(self.thrink1(d3)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.upsample3(self.senet6(self.thrink3(us2)))
        return u3


class DenseStack2(nn.Module):
    dump_patches = True

    def __init__(self, input_channel, output_channel, final_upsample=True, ret_mid=False):
        super(DenseStack2, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel * 4)
        self.senet3 = SenetBlock(input_channel * 8, 8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel * 8)
        self.dense5 = DenseBlock2_noExpand(input_channel * 8)
        self.thrink1 = nn.Sequential(mobile_unit(input_channel * 8, input_channel * 4, num3x3=1), mobile_unit(input_channel * 4, input_channel * 4, num3x3=2))
        self.senet4 = SenetBlock(input_channel * 4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink2 = nn.Sequential(mobile_unit(input_channel * 4, input_channel * 2, num3x3=1), mobile_unit(input_channel * 2, input_channel * 2, num3x3=2))
        self.senet5 = SenetBlock(input_channel * 2, 8)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.thrink3 = nn.Sequential(mobile_unit(input_channel * 2, input_channel * 2, num3x3=1), mobile_unit(input_channel * 2, output_channel, num3x3=2))
        self.senet6 = SenetBlock(output_channel, 16)
        self.final_upsample = final_upsample
        if self.final_upsample:
            self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ret_mid = ret_mid

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.senet3(self.dense3(d2)))
        d4 = self.dense5(self.dense4(d3))
        u1 = self.upsample1(self.senet4(self.thrink1(d4)))
        us1 = d2 + u1
        u2 = self.upsample2(self.senet5(self.thrink2(us1)))
        us2 = d1 + u2
        u3 = self.senet6(self.thrink3(us2))
        if self.final_upsample:
            u3 = self.upsample3(u3)
        if self.ret_mid:
            mid_feats = [d1, d2, d3, d4]
            return u3, mid_feats
        else:
            return u3, d4


class DenseStack2_Encoder(nn.Module):
    dump_patches = True

    def __init__(self, input_channel):
        super(DenseStack2_Encoder, self).__init__()
        self.dense1 = DenseBlock2(input_channel)
        self.senet1 = SenetBlock(input_channel * 2, 32)
        self.transition1 = nn.AvgPool2d(2)
        self.dense2 = DenseBlock(input_channel * 2)
        self.senet2 = SenetBlock(input_channel * 4, 16)
        self.transition2 = nn.AvgPool2d(2)
        self.dense3 = DenseBlock(input_channel * 4)
        self.senet3 = SenetBlock(input_channel * 8, 8)
        self.transition3 = nn.AvgPool2d(2)
        self.dense4 = DenseBlock2_noExpand(input_channel * 8)
        self.dense5 = DenseBlock2_noExpand(input_channel * 8)

    def forward(self, x):
        d1 = self.transition1(self.senet1(self.dense1(x)))
        d2 = self.transition2(self.senet2(self.dense2(d1)))
        d3 = self.transition3(self.senet3(self.dense3(d2)))
        d4 = self.dense5(self.dense4(d3))

        return d4


class H2ONet_Backnone(nn.Module):

    def __init__(self, cfg, input_channel=128, out_channel=24, latent_size=256, kpts_num=21):
        """Init a DenseStack

        Args:
            input_channel (int, optional): the first-layer channel size. Defaults to 128.
            out_channel (int, optional): output channel size. Defaults to 24.
            latent_size (int, optional): middle-feature channel size. Defaults to 256.
            kpts_num (int, optional): amount of 2D landmark. Defaults to 21.
            pretrain (bool, optional): use pretrain weight or not. Defaults to True.
        """
        super(H2ONet_Backnone, self).__init__()
        self.cfg = cfg
        self.pre_layer = nn.Sequential(conv_layer(3, input_channel // 2, 3, 2, 1), mobile_unit(input_channel // 2, input_channel))
        self.thrink = conv_layer(input_channel * 4, input_channel)
        self.dense_stack1 = DenseStack(input_channel, out_channel)
        self.stack1_remap = conv_layer(out_channel, out_channel)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.thrink2 = conv_layer((out_channel + input_channel), input_channel)
        self.dense_stack2 = DenseStack2(input_channel, out_channel, final_upsample=False)
        self.dense_stack3 = DenseStack2_Encoder(input_channel)

        self.mid_proj = conv_layer(1024, latent_size, 1, 1, 0, bias=False, bn=False, relu=False)
        self.reduce = conv_layer(out_channel, kpts_num, 1, bn=False, relu=False)
        self.uv_reg = nn.Sequential(linear_layer(latent_size, 128, bn=False), linear_layer(128, 64, bn=False), linear_layer(64, 2, bn=False, relu=False))
        self.reorg = Reorg()
        if self.cfg.model.pretrain:
            weight = torch.load("model/mob_recon/densestack.pth")
            self.load_state_dict(weight, strict=False)
            logger.info("Load pre-trained weight: densestack.pth")

    def forward(self, x):
        pre_out = self.pre_layer(x)  # Conv3*3 + Mobile unit, output_c = input_c
        pre_out_reorg = self.reorg(pre_out)  # Channel shuffle
        thrink = self.thrink(pre_out_reorg)  # Conv1*1, output_c = input_c // 4
        stack1_out = self.dense_stack1(thrink)
        stack1_out_remap = self.stack1_remap(stack1_out)  # Conv1*1, output_c = input_c
        input2 = torch.cat((stack1_out_remap, thrink), dim=1)
        thrink2 = self.thrink2(input2)
        stack2_out, stack2_mid = self.dense_stack2(thrink2)
        latent = self.mid_proj(stack2_mid)
        uv_reg = self.uv_reg(self.reduce(stack2_out).view(stack2_out.shape[0], 21, -1))
        stack3_mid = self.dense_stack3(thrink2.detach())

        return latent, stack2_mid, stack3_mid, uv_reg
