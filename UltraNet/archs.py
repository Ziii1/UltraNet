import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import paddle
import cv2
from einops import rearrange, repeat

__all__ = ['UltraNet']


class FoFo(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FoFo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ShalFoFo(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = FoFo(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class DeS(nn.Module):
    def __init__(self, dim_in, dim_out, filters, kernel_size, padding, stride=2, dilation=1, ):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                               groups=dim_in)

        self.norm_layer = nn.GroupNorm(num_groups=dim_in, num_channels=dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm_layer(x)
        return self.conv2(x)


class DeSFoFo(nn.Module):
    def __init__(self, in_c, out_c, filters, stride=2, kernel_size=3):
        super().__init__()

        self.w = nn.Sequential(
            DeS(in_c, out_c, kernel_size, stride, filters),
            nn.GELU()
        )
        self.z = nn.Conv2d(in_c, out_c, 3, padding=1, stride=2, )
        self.attn = FoFo(out_c)

    def forward(self, inputs):
        x = self.w(inputs)
        z = self.z(inputs)
        y = self.attn(x + z)
        return y


class DeKFoFo(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[7, 5, 2, 1]):
        super().__init__()

        self.mda0 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[0] - 1)) // 2,
                              dilation=dilated_ratio[0], groups=in_c // 4)
        self.mda1 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[1] - 1)) // 2,
                              dilation=dilated_ratio[1], groups=in_c // 4)
        self.mda2 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[2] - 1)) // 2,
                              dilation=dilated_ratio[2], groups=in_c // 4)
        self.mda3 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[3] - 1)) // 2,
                              dilation=dilated_ratio[3], groups=in_c // 4)
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=2)

        self.FoFo = FoFo(out_c)

    def forward(self, x):
        x = torch.chunk(x, 4, dim=1)
        x0 = self.mda0(x[0])
        x1 = self.mda1(x[1])
        x2 = self.mda2(x[2])
        x3 = self.mda3(x[3])
        x = self.conv(self.norm_layer(torch.cat((x0, x1, x2, x3), dim=1)))
        y = self.FoFo(x)
        return y


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=8, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]  # 获取batch_size

        feats = [conv(x) for conv in self.convs]  # 让x分成3*3和5*5进行卷积
        feats = torch.cat(feats, dim=1)  # 合并卷积结果
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        # reshape一下大小
        # 接下来计算图中的U
        feats_U = torch.sum(feats, dim=1)  # 两个分支得到的卷积结果相加
        feats_S = self.gap(feats_U)  # 自适应池化，也就是对各个chanel求均值得到图中的S
        feats_Z = self.fc(feats_S)  # fc层压缩特征得到图中的Z

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # 不同的分支各自恢复特征Z到channel的宽度
        attention_vectors = torch.cat(attention_vectors, dim=1)  # 连接起来方便后续操作
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        # reshape起来方便后续操作
        attention_vectors = self.softmax(attention_vectors)  # softmax得到图中的a和b

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        # 把softmax后的各自自注意力跟卷积后的结果相乘，得到图中select的结果，然后相加得到最终输出

        return feats_V


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):  # 6 12 18 | 3 5 7 | 4 8 12
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UltraNet(nn.Module):
    def __init__(self, in_c, out_c, n_classes=1):
        super(UltraNet, self).__init__()

        self.c1 = ShalFoFo(3, 8, stride=1)
        self.c2 = ShalFoFo(8, 16, stride=2)
        # The first branch
        self.c13 = DeSFoFo(16, 24, filters=16)
        self.c14 = DeSFoFo(24, 32, filters=24)
        self.c15 = DeSFoFo(32, 48, filters=32)
        self.c16 = DeSFoFo(48, 64, filters=48)

        # The second branch
        self.c23 = DeKFoFo(16, 24)
        self.c24 = DeKFoFo(24, 32)
        self.c25 = DeKFoFo(32, 48)
        self.c26 = DeKFoFo(48, 64)

        self.channel_attention3 = channel_attention(24)
        self.spatial_attention3 = spatial_attention(3)
        self.channel_attention4 = channel_attention(32)
        self.spatial_attention4 = spatial_attention(3)
        self.channel_attention5 = channel_attention(48)
        self.spatial_attention5 = spatial_attention(3)
        self.channel_attention6 = channel_attention(64)
        self.spatial_attention6 = spatial_attention(3)

        self.aspp1 = ASPP(8, 8)
        self.aspp2 = ASPP(8, 3)
        self.output = nn.Conv2d(3, 1, kernel_size=1, padding=0)

        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

        ## -------------Decoder--------------
        '''stage 5d'''

        self.hd6_UT_hd5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd6_UT_hd5_conv = nn.Conv2d(64, 48, 3, padding=1)
        self.hd6_UT_hd5_bn = nn.BatchNorm2d(48)
        self.hd6_UT_hd5_relu = nn.ReLU(inplace=True)

        '''stage 4d'''

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = nn.Conv2d(48, 32, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(32)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        '''stage 3d'''

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = nn.Conv2d(32, 24, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(24)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        '''stage 2d '''

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv2d(24, 16, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(16)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        '''stage 1d'''

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd2_UT_hd1_conv = nn.Conv2d(16, 8, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(8)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.sk1 = SKConv(64)
        self.sk2 = SKConv(48)
        self.sk3 = SKConv(32)
        self.sk4 = SKConv(24)
        self.sk5 = SKConv(16)
        self.sk6 = SKConv(8)


    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)

        c13 = self.c13(c2)
        c14 = self.c14(c13)
        c15 = self.c15(c14)
        c16 = self.c16(c15)

        c23 = self.c23(c2)
        c24 = self.c24(c23)
        c25 = self.c25(c24)
        c26 = self.c26(c25)

        att3 = ((self.channel_attention3(c13) + self.spatial_attention3(c13)) * c23) + c13
        att4 = ((self.channel_attention4(c14) + self.spatial_attention4(c14)) * c24) + c14
        att5 = ((self.channel_attention5(c15) + self.spatial_attention5(c15)) * c25) + c15
        att6 = ((self.channel_attention6(c16) + self.spatial_attention6(c16)) * c26) + c16

        ## -------------Decoder------------
        hd6 = self.sk1(att6)
        hd5 = self.sk2(self.hd6_UT_hd5_relu(self.hd6_UT_hd5_bn(self.hd6_UT_hd5_conv(self.hd6_UT_hd5(hd6)))) + att5)
        hd4 = self.sk3(self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))) + att4)
        hd3 = self.sk4(self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))) + att3)
        hd2 = self.sk5(self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))) + c2)
        hd1 = self.sk6(self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))) + c1)

        output = self.aspp2(self.aspp1(hd1))
        output = self.output(output)

        return output
