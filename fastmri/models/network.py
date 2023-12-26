import torch.nn as nn
import torch
from fastmri.models import common
from torch.nn import functional as F


class unet(torch.nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(True))
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(True))
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True))
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ReLU(True))
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(True))

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x3 = x2+x3
        x4 = self.decoder1(x3)
        x4 = x4+x1
        x5 = self.decoder2(x4)
        return x5

class Concat(torch.nn.Module):
    def __init__(self, channel):
        super(Concat, self).__init__()
        self.conv = nn.Conv2d(channel*2, channel, 3, 1, 1)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x, x

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

# class Branch_fusion(torch.nn.Module):
#     def __init__(self, channel):
#         super(Branch_fusion, self).__init__()
#         self.conv = nn.Conv2d(channel * 2, channel, 3, 1, 1)
#         self.mscam1 = MS_CAM(channels=channel)
#         self.mscam2 = MS_CAM(channels=channel)
#
#     def forward(self, x1, x2):
#         x1 = self.mscam1(x1)
#         x2 = self.mscam2(x2)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.conv(x)
#         x1 = x1+x
#         x2 = x2+x
#
#         return x1, x2

class Branch_fusion(torch.nn.Module):
    def __init__(self, channels, r=4):
        super(Branch_fusion, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(channels*2, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels*2, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels*2, channels, 3, 1, 1)

    def forward(self, x1, x2):
        x1l = self.local_att(x1)
        x1g = self.global_att(x1)

        x2l = self.local_att2(x2)
        x2g = self.global_att2(x2)

        w1 = self.sigmoid(x1l + x1g)
        w2 = self.sigmoid(x2l + x2g)

        x1 = w1*x1
        x2 = w2*x2

        # xl = x1l+x2l
        # xg = x1g+x2g
        #
        # wl = self.sigmoid(xl)
        # wg = self.sigmoid(xg)
        #
        # xll = x1l*wl+x2l*(1-wl)
        # xgg = x1g*wg+x2g*(1-wg)
        #
        # # x = torch.cat([xll, xgg], dim=1)
        # x = xll+xgg
        # x = self.conv(x)

        xl = torch.cat([x1l, x2l], dim=1)
        xg = torch.cat([x1g, x2g], dim=1)

        xl = self.conv1(xl)
        xg = self.conv2(xg)

        x = xl+xg
        # x = torch.cat([xl, xg], dim=1)
        # x = self.conv3(x)

        x1 = x+x1
        x2 = x+x2

        return x1, x2
    
# class Branch_fusion(torch.nn.Module):
#     def __init__(self, channels, r=4):
#         super(Branch_fusion, self).__init__()
#
#         self.conv = nn.Conv2d(channels*2, channels, 3, 1, 1)
#
#     def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim=1)
#         x = self.conv(x)
#         return x, x


class Modal_fusion(torch.nn.Module):
    def __init__(self, channel):
        super(Modal_fusion, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel, channel, 5, 1, 2)

        self.s = nn.Sigmoid()

        self.conv = nn.Conv2d(channel*2, channel, 3, 1, 1)


    def forward(self, x1, x2):
        x1_1 = self.conv1(x1)
        x1_2 = self.conv2(x1)
        x1_3 = self.conv3(x1)

        x2_1 = torch.cat([x1_1, x2], dim=1)
        x2_2 = torch.cat([x1_2, x2], dim=1)
        x2_3 = torch.cat([x1_3, x2], dim=1)

        w1 = self.s(x2_1)
        w2 = self.s(x2_2)
        w3 = self.s(x2_3)

        x = x2_1*w1+x2_2*w2+x2_3*w3

        x = self.conv(x)

        return x


# # 没有辅助分支
# class net(torch.nn.Module):
#     def __init__(self, conv=common.default_conv):
#         super(net, self).__init__()
#         n_colors = 1
#         n_feats = 64
#         scale = 4
#         kernel_size = 3
#         act = nn.ReLU(True)
#
#         self.head1 = conv(n_colors, n_feats, kernel_size)
#         self.head2 = conv(n_colors, n_feats, kernel_size)
#
#         self.ResBlock11 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock21 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#
#         self.ResBlock12 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock22 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#
#         self.ResBlock13 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock23 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         # self.ResBlock33 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         # self.ResBlock14 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         #self.ResBlock24 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         # self.ResBlock15 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         #self.ResBlock25 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#
#         self.ResBlock1 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock2 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock3 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock4 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock5 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock6 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#
#         # self.unet1 = unet()
#         # self.unet2 = unet()
#         # self.unet3 = unet()
#         # self.unet4 = unet()
#         # self.unet5 = unet()
#         # self.unet6 = unet()
#
#         self.unet1 = Unet()
#         self.unet2 = Unet()
#         self.unet3 = Unet()
#         self.unet4 = Unet()
#         self.unet5 = Unet()
#         self.unet6 = Unet()
#
#         m_tail1 = [
#             common.Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, n_colors, kernel_size)
#         ]
#         self.tail1 = nn.Sequential(*m_tail1)
#
#         m_tail2 = [
#             common.Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, n_colors, kernel_size)
#         ]
#         self.tail2 = nn.Sequential(*m_tail2)
#
#         self.tail3 = nn.Conv2d(64, 1, 3, 1, 1)
#
#         self.bf1 = Branch_fusion(channels=64)
#         self.bf2 = Branch_fusion(channels=64)
#         self.bf3 = Branch_fusion(channels=64)
#         self.bf4 = Branch_fusion(channels=64)
#         self.bf5 = Branch_fusion(channels=64)
#         self.bf6 = Branch_fusion(channels=64)
#
#     def forward(self, x1, x2):
#         x1 = x1.float()
#         x2 = x2.float()
#         x2 = self.head2(x2)    # 初始特征 要加到最后
#
#         #x1_1 = self.ResBlock11(x1)
#         x2_1 = self.ResBlock21(x2)
#         #x2_1 = x2_1+x1_1
#         #x2_1 = self.cat1(x2_1, x1_1)
#         #x2_1 = self.mf1(x1_1, x2_1)
#
#         #x1_2 = self.ResBlock12(x1_1)
#         x2_2 = self.ResBlock22(x2_1)
#         #x2_2 = x2_2+x1_2
#         #x2_2 = self.cat2(x2_2, x1_2)
#         #x2_2 = self.mf2(x1_2, x2_2)
#
#         #x1_3 = self.ResBlock13(x1_2)
#         x2_3 = self.ResBlock23(x2_2)
#         # # x2_2 = x2_2+x1_2
#         # # x2_2 = self.cat2(x2_2, x1_2)
#         #x2_3 = self.mf3(x1_3, x2_3)
#
#         #x1_3 = self.ResBlock13(x1_2)
#         #x2_3 = self.ResBlock23(x2_2)
#         #x2_3 = self.mf3(x1_3, x2_3)
#
#         #x1_4 = self.ResBlock14(x1_3)
#         #x2_4 = self.ResBlock24(x2_3)
#         #x2_4 = self.mf4(x1_4, x2_4)
#
#         #x1_5 = self.ResBlock15(x1_4)
#         #x2_5 = self.ResBlock25(x2_4)
#         #x2_5 = self.mf5(x1_5, x2_5)
#
#         # out1 = x2_3+x2
#         # out1 = self.tail1(out1)
#
#         re1 = self.unet1(x2_3)
#         sr1 = self.ResBlock1(x2_3)
#         #sr1 = sr1+re1
#         #sr1, re1 = self.cat1(sr1, re1)
#         sr1, re1 = self.bf1(sr1, re1)
#
#         re2 = self.unet2(re1)
#         sr2 = self.ResBlock2(sr1)
#         #sr2 =
#         #sr2, re2 = self.cat2(sr2, re2)
#         sr2, re2 = self.bf2(sr2, re2)
#
#         re3 = self.unet3(re2)
#         sr3 = self.ResBlock3(sr2)
#         # sr2 =
#         #sr3, re3 = self.cat3(sr3, re3)
#         sr3, re3 = self.bf3(sr3, re3)
#
#         re4 = self.unet4(re3)
#         sr4 = self.ResBlock4(sr3)
#         #sr4, re4 = self.cat4(sr4, re4)
#         sr4, re4 = self.bf4(sr4, re4)
#
#         re5 = self.unet5(re4)
#         sr5 = self.ResBlock5(sr4)
#         #sr5, re5 = self.cat5(sr5, re5)
#         sr5, re5 = self.bf5(sr5, re5)
#
#         re6 = self.unet6(re5)
#         sr6 = self.ResBlock6(sr5)
#         #sr6, re6 = self.cat6(sr6, re6)
#         sr6, re6 = self.bf6(sr6, re6)
#
#         out2 = re6+x2
#         out2 = self.tail3(out2)
#
#         out3 = sr6+x2
#         out3 = self.tail2(out3)
#
#         return out2, out3  #out1,

class net(torch.nn.Module):
    def __init__(self, conv=common.default_conv):
        super(net, self).__init__()
        n_colors = 1
        n_feats = 64
        scale = 4
        kernel_size = 3
        act = nn.ReLU(True)

        self.head1 = conv(n_colors, n_feats, kernel_size)
        self.head2 = conv(n_colors, n_feats, kernel_size)

        self.ResBlock11 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock21 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)

        self.ResBlock12 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock22 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)

        self.ResBlock13 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock23 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        # self.ResBlock33 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        # self.ResBlock14 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        #self.ResBlock24 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        # self.ResBlock15 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        #self.ResBlock25 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)

        self.ResBlock1 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock2 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock3 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock4 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock5 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
        self.ResBlock6 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)

        # self.unet1 = unet()
        # self.unet2 = unet()
        # self.unet3 = unet()
        # self.unet4 = unet()
        # self.unet5 = unet()
        # self.unet6 = unet()

        self.unet1 = Unet()
        self.unet2 = Unet()
        self.unet3 = Unet()
        self.unet4 = Unet()
        self.unet5 = Unet()
        self.unet6 = Unet()

        m_tail1 = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.tail1 = nn.Sequential(*m_tail1)

        m_tail2 = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.tail2 = nn.Sequential(*m_tail2)

        self.tail3 = nn.Conv2d(64, 1, 3, 1, 1)

        # self.cat1 = Concat(channel=64)
        # self.cat2 = Concat(channel=64)
        # self.cat3 = Concat(channel=64)
        # self.cat4 = Concat(channel=64)
        # self.cat5 = Concat(channel=64)
        # self.cat6 = Concat(channel=64)

        self.mf1 = Modal_fusion(channel=64)
        self.mf2 = Modal_fusion(channel=64)
        self.mf3 = Modal_fusion(channel=64)
        # self.mf4 = Modal_fusion(channel=64)
        # self.mf5 = Modal_fusion(channel=64)

        self.bf1 = Branch_fusion(channels=64)
        self.bf2 = Branch_fusion(channels=64)
        self.bf3 = Branch_fusion(channels=64)
        self.bf4 = Branch_fusion(channels=64)
        self.bf5 = Branch_fusion(channels=64)
        self.bf6 = Branch_fusion(channels=64)

    def forward(self, x1, x2):
        x1 = x1.float()
        x2 = x2.float()
        x1 = self.head1(x1)
        x2 = self.head2(x2)    # 初始特征 要加到最后

        x1_1 = self.ResBlock11(x1)
        x2_1 = self.ResBlock21(x2)
        #x2_1 = x2_1+x1_1
        #x2_1 = self.cat1(x2_1, x1_1)
        x2_1 = self.mf1(x1_1, x2_1)
        f1 = x2_1

        x1_2 = self.ResBlock12(x1_1)
        x2_2 = self.ResBlock22(x2_1)
        #x2_2 = x2_2+x1_2
        #x2_2 = self.cat2(x2_2, x1_2)
        x2_2 = self.mf2(x1_2, x2_2)
        f2 = x2_2

        x1_3 = self.ResBlock13(x1_2)
        x2_3 = self.ResBlock23(x2_2)
        # # x2_2 = x2_2+x1_2
        # # x2_2 = self.cat2(x2_2, x1_2)
        x2_3 = self.mf3(x1_3, x2_3)
        f3 = x2_3

        #x1_3 = self.ResBlock13(x1_2)
        #x2_3 = self.ResBlock23(x2_2)
        #x2_3 = self.mf3(x1_3, x2_3)

        #x1_4 = self.ResBlock14(x1_3)
        #x2_4 = self.ResBlock24(x2_3)
        #x2_4 = self.mf4(x1_4, x2_4)

        #x1_5 = self.ResBlock15(x1_4)
        #x2_5 = self.ResBlock25(x2_4)
        #x2_5 = self.mf5(x1_5, x2_5)

        # out1 = x2_3+x2
        # out1 = self.tail1(out1)

        re1 = self.unet1(x2_3)
        sr1 = self.ResBlock1(x2_3)
        #sr1 = sr1+re1
        #sr1, re1 = self.cat1(sr1, re1)
        sr1, re1 = self.bf1(sr1, re1)

        re2 = self.unet2(re1)
        sr2 = self.ResBlock2(sr1)
        #sr2 =
        #sr2, re2 = self.cat2(sr2, re2)
        sr2, re2 = self.bf2(sr2, re2)

        re3 = self.unet3(re2)
        sr3 = self.ResBlock3(sr2)
        # sr2 =
        #sr3, re3 = self.cat3(sr3, re3)
        sr3, re3 = self.bf3(sr3, re3)

        re4 = self.unet4(re3)
        sr4 = self.ResBlock4(sr3)
        #sr4, re4 = self.cat4(sr4, re4)
        sr4, re4 = self.bf4(sr4, re4)

        re5 = self.unet5(re4)
        sr5 = self.ResBlock5(sr4)
        #sr5, re5 = self.cat5(sr5, re5)
        sr5, re5 = self.bf5(sr5, re5)

        re6 = self.unet6(re5)
        sr6 = self.ResBlock6(sr5)
        #sr6, re6 = self.cat6(sr6, re6)
        sr6, re6 = self.bf6(sr6, re6)

        out2 = re6+x2
        out2 = self.tail3(out2)

        out3 = sr6+x2
        out3 = self.tail2(out3)

        return out2, out3#, f1, f2, f3, sr1, re1, sr2, re2, sr3, re3, sr4, re4, sr5, re5, sr6, re6  #out1,


# # 没有重建分支
# class net(torch.nn.Module):
#     def __init__(self, conv=common.default_conv):
#         super(net, self).__init__()
#         n_colors = 1
#         n_feats = 64
#         scale = 4
#         kernel_size = 3
#         act = nn.ReLU(True)
# 
#         self.head1 = conv(n_colors, n_feats, kernel_size)
#         self.head2 = conv(n_colors, n_feats, kernel_size)
# 
#         self.ResBlock11 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock21 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
# 
#         self.ResBlock12 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock22 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
# 
#         self.ResBlock13 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock23 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
# 
#         self.ResBlock1 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock2 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock3 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock4 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock5 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
#         self.ResBlock6 = common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
# 
#         m_tail2 = [
#             common.Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, n_colors, kernel_size)
#         ]
#         self.tail2 = nn.Sequential(*m_tail2)
# 
# 
#         self.mf1 = Modal_fusion(channel=64)
#         self.mf2 = Modal_fusion(channel=64)
#         self.mf3 = Modal_fusion(channel=64)
# 
#     def forward(self, x1, x2):
#         x1 = x1.float()
#         x2 = x2.float()
#         x1 = self.head1(x1)
#         x2 = self.head2(x2)    # 初始特征 要加到最后
# 
#         x1_1 = self.ResBlock11(x1)
#         x2_1 = self.ResBlock21(x2)
#         x2_1 = self.mf1(x1_1, x2_1)
# 
#         x1_2 = self.ResBlock12(x1_1)
#         x2_2 = self.ResBlock22(x2_1)
#         x2_2 = self.mf2(x1_2, x2_2)
# 
#         x1_3 = self.ResBlock13(x1_2)
#         x2_3 = self.ResBlock23(x2_2)
#         x2_3 = self.mf3(x1_3, x2_3)
# 
#         sr1 = self.ResBlock1(x2_3)
# 
#         sr2 = self.ResBlock2(sr1)
# 
#         sr3 = self.ResBlock3(sr2)
# 
#         sr4 = self.ResBlock4(sr3)
# 
#         sr5 = self.ResBlock5(sr4)
# 
#         sr6 = self.ResBlock6(sr5)
# 
#         out3 = sr6+x2
#         out3 = self.tail2(out3)
# 
#         return out3, out3  



class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int = 64,
        out_chans: int = 64,
        chans: int = 128,
        num_pool_layers: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)