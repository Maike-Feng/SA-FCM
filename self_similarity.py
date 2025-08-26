import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from einops import rearrange, repeat, reduce


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc):
        super(GlobalHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):  # [64, 30, 11, 11]
        x = self.pool(x)  # [64, 30, 11, 11]->[64, 30, 1, 1]
        x = x.view(x.size(0), -1)  # [64, 30]
        x = self.fc(x)  # torch.Size([64, 64])
        return x


class SelfCorrelationComputation(nn.Module):
    def __init__(self, unfold_size=5):
        super(SelfCorrelationComputation, self).__init__()
        self.unfold_size = (unfold_size, unfold_size)  # 7
        self.padding_size = unfold_size // 2  # 3
        # nn.Unfold()输出形状为(batch_size, channels*kernel_size[0]*kernel_size[1], L)
        # 其中, channels*kernel_size[0]*kernel_size[1] 是每个通道的卷积核大小，L 是展开后每个通道的空间位置数量
        self.unfold = nn.Unfold(kernel_size=self.unfold_size, padding=self.padding_size)

    def forward(self, q):
        b, c, h, w = q.shape  # [2, 256, 11, 11]
        q = F.normalize(q, dim=1, p=2)  # 将某一个维度除以那个维度对应的范数 torch.Size([2, 256, 11, 11])  # 先除2范数
        q_unfold = self.unfold(q)  # torch.Size([2, 12544=256*7*7, 121=11*11])
        q_unfold = q_unfold.view(b, c, self.unfold_size[0], self.unfold_size[1], h, w) # b, c, u, v, h, w  torch.Size([2, 256, 7, 7, 11, 11])
        self_sim = q_unfold * q.unsqueeze(2).unsqueeze(2)  # (2, 256, 7, 7, 11, 11) * (2, 256, 1, 1, 11, 11)->[2, 256, 7, 7, 11, 11] # 逐通道的计算,每一个像素再乘以四围7*7的像素
        self_sim = self_sim.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v  torch.Size([2, 256, 11, 11, 7, 7])

        return self_sim


class SelfSimilarityEncoder(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size, ksize):
        super(SelfSimilarityEncoder, self).__init__()
        def make_building_conv_block(in_channel, out_channel, ksize, padding=(0,0,0), stride=(1,1,1), bias=True, conv_group=1):
            building_block_layers = []
            building_block_layers.append(nn.Conv3d(in_channel, out_channel, (1, ksize, ksize), stride=stride, bias=bias, groups=conv_group, padding=padding))
            building_block_layers.append(nn.BatchNorm3d(out_channel))
            building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        conv_in_block_list = [make_building_conv_block(mid_ch, mid_ch, ksize) for _ in range(unfold_size//ksize)]
        self.conv_in = nn.Sequential(*conv_in_block_list)
        self.conv1x1_out = nn.Sequential(nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=True, padding=0),
                                         nn.BatchNorm2d(in_ch))

    def forward(self, x):  # [2, 30, 11, 11, 7, 7]  torch.Size([64, 128, 11, 11, 7, 7])
        b, c, h, w, u, v = x.shape

        x = x.view(b, c, h * w, u, v)  # [2, 30, 11*11, 7, 7]
        x = self.conv_in(x)
        c = x.shape[1]
        x = x.mean(dim=[-1,-2]).view(b, c, h, w)  # [2, 30, 11, 11]

        x = self.conv1x1_out(x)  # [2, 30, 11, 11]  线性层使通道大小恢复到原始特征映射F的通道大小

        return x


class SSM(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size=7, ksize=3, nc=16):
        super(SSM, self).__init__()
        self.ch_reduction_encoder = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False, padding=0)  # 线性层
        self.SCC = SelfCorrelationComputation(unfold_size=unfold_size)
        self.SSE = SelfSimilarityEncoder(in_ch, mid_ch, unfold_size=unfold_size, ksize=ksize)

        self.FFN = nn.Sequential(
                                nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=2, bias=True, padding=1),
                                nn.BatchNorm2d(in_ch),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, bias=True, padding=0),
                                nn.BatchNorm2d(in_ch),
                                GlobalHead(in_ch, nc=nc)
                                )

        # ----------------------------- Sub-pixel Convolution --------------------------------
        self.upscale_factor = 3
        self.sub_pixel_conv_1 = nn.Sequential(
                                            nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2, bias=True),
                                            nn.BatchNorm2d(in_ch),
                                            nn.ReLU(inplace=True)
                                            )
        self.sub_pixel_conv_2 = nn.Sequential(
                                            nn.Conv2d(in_ch, in_ch * (self.upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                            nn.BatchNorm2d(in_ch * (self.upscale_factor ** 2))
                                            )
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        
        self.r_conv = nn.Sequential(
                                    nn.Conv3d(1, 1, kernel_size=(1, self.upscale_factor, self.upscale_factor), stride=(1, self.upscale_factor, self.upscale_factor), bias=True, groups=1, padding=0),
                                    nn.BatchNorm3d(1),
                                    nn.ReLU(inplace=True)
                                    )
        
        self.batch_norm_end = nn.BatchNorm2d(in_ch)


    def forward(self, ssm_input_feat):  # torch.Size([8, 32, 11, 11])
        """"""
        q = self.ch_reduction_encoder(ssm_input_feat)  # 通过线性层降低光谱维度  # torch.Size([8, 128, 11, 11])
        self_sim = self.SCC(q)  # torch.Size([8, 128, 11, 11, 7, 7])
        self_sim_feat = self.SSE(self_sim)  # torch.Size([8, 32, 11, 11]) 输出大小与输入大小相等
        # ----------------------------- Sub-pixel Convolution --------------------------------
        sub_pixel_conv_out_1 = self.sub_pixel_conv_1(ssm_input_feat)
        sub_pixel_conv_out_2 = self.sub_pixel_conv_2(sub_pixel_conv_out_1)
        pixel_shuffle_out = self.pixel_shuffle(sub_pixel_conv_out_2)
        conv_pixel_shuffle = self.r_conv(pixel_shuffle_out.unsqueeze(1)).squeeze(1)  # torch.Size([32, 1, 64, 15, 15])

        
        ssm_output_feat = (0.4*conv_pixel_shuffle + 0.6*self_sim_feat) + self.batch_norm_end(ssm_input_feat)  #  torch.Size([8, 32, 11, 11])
        ssm_output_feat = self.FFN(ssm_output_feat)  # torch.Size([8, 16])

        
        return ssm_output_feat


if __name__ == '__main__':
    data = torch.randn(32, 32, 15, 15).cuda()  # 输入形状 IP
    model = SSM(in_ch=32, mid_ch=32, unfold_size=11, ksize=11, nc=16).cuda()
    
    # data = torch.randn(32, 19, 15, 15).cuda()  # 输入形状 PU
    # model = SSM(in_ch=19, mid_ch=32, unfold_size=9, ksize=9, nc=9).cuda()
    
    # data = torch.randn(32, 32, 15, 15).cuda()  # 输入形状 HU
    # model = SSM(in_ch=32, mid_ch=32, unfold_size=5, ksize=5, nc=15).cuda()

    # data = torch.randn(32, 32, 15, 15).cuda()  # 输入形状 HanChuan
    # model = SSM(in_ch=32, mid_ch=32, unfold_size=7, ksize=7, nc=16).cuda()
    
    # data = torch.randn(32, 32, 15, 15).cuda()  # 输入形状 Trento
    # model = SSM(in_ch=32, mid_ch=32, unfold_size=7, ksize=7, nc=6).cuda()
    import time
    start = time.time()
    model_result = model(data)  # torch.Size([32, 16])
    print('time:', time.time()-start)
    print(model_result.shape)
    # summary(model, [(32, 15, 15)])

    num=0
    for i in model.parameters():
        if i.requires_grad==True:
            num+=i.numel()
    print(num)
    
    # import scipy.io as sio
    # import numpy as np
    # label = sio.loadmat('/mnt/d2/FengJiajie/datasets/Trento/Trento_gt.mat')['Trento_gt']
    # data = sio.loadmat('/mnt/d2/FengJiajie/datasets/Trento/Trento.mat')['Trento']
    # print(data.shape)  # (166, 600, 63)
    # print(np.max(label),np.min(label))  # (166, 600) 6,0