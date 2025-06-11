import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import deform_conv2d
from einops import rearrange


class ModulatedDeformableConv2d(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deformable_groups=1,
                 extra_offset_mask=True,
                 offset_in_channel=32
                 ):
        super(ModulatedDeformableConv2d, self).__init__()

        self.in_channel = channel
        self.out_channel = channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.extra_offset_mask = extra_offset_mask

        self.conv_offset_mask = nn.Conv2d(offset_in_channel,
                                          deformable_groups * 3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=self.padding,
                                          bias=True)

        self.init_offset()

        self.weight = nn.Parameter(torch.Tensor(self.out_channel, self.in_channel // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channel))
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        n = self.in_channel * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        torch.nn.init.constant_(self.conv_offset_mask.weight, 0.)
        torch.nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        if self.extra_offset_mask:
            offset_mask = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        out = deform_conv2d(input=x,
                            offset=offset,
                            weight=self.weight,
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            mask=mask
                            )

        return out


class ResBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class MDC(nn.Module):
    def __init__(self, channel, offset_channel):
        super().__init__()
        self.dcnpack = ModulatedDeformableConv2d(channel, 3, stride=1, padding=1, dilation=1,
                                                 deformable_groups=8, extra_offset_mask=True, offset_in_channel=offset_channel)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, offset):
        x = self.lrelu(self.dcnpack([x, offset]))
        return x


class ADC(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv0 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=channel)
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=1)

        self.conv3 = nn.Conv2d(channel, channel * kernel_size * kernel_size, 1, 1, 0)

        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fusion = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        pre = x
        x0 = self.conv0(x)

        b, c, h, w = x.size()
        x1 = self.conv1(x0)
        x2 = self.avg_pool(x0)
        x2 = self.conv2(x2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = x1 + x2
        out = self.conv3(out)

        filter_x = out.reshape([b, c, self.kernel_size * self.kernel_size, h, w])
        unfold_x = self.unfold(x0).reshape(b, c, -1, h, w)
        out = (unfold_x * filter_x).sum(2)

        out = self.lrelu(out)
        out = self.fusion(out) + pre
        return out


class MRAB(nn.Module):
    def __init__(self, input_channel, output_channel, offset_channel):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)

        self.mdc = MDC(output_channel, offset_channel)
        self.adc = ADC(output_channel)

        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        x = self.mdc(x, offset)
        out = self.adc(x) + x

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class OffsetBlock(nn.Module):
    def __init__(self, input_channel, offset_channel, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel * 2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class HCB(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
        self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.conv3_1 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
        self.conv2_1 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
        self.conv1_1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
        self.fusion = nn.Conv2d(output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        dc1 = self.conv1(x_reduce)
        dc1_1 = self.lrelu(dc1)
        dc2 = self.conv2(dc1_1)
        dc2_1 = self.lrelu(dc2)
        dc3 = self.conv3(dc2_1)
        dc3_1 = self.lrelu(dc3)
        dc4 = self.conv4(dc3_1)
        dc4 = self.lrelu(dc4)
        dc5 = self.conv3_1(dc4 + dc3)
        dc5 = self.lrelu(dc5)
        dc6 = self.conv2_1(dc5 + dc2)
        dc6 = self.lrelu(dc6)
        conv7 = self.lrelu(self.conv1_1(dc6 + dc1))
        out = self.fusion(conv7) + x
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([x, y], dim=1)
        return x


class AFAM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)

        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):

        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, n_channel=32, offset_channel=32):
        super(UNet, self).__init__()

        dim = n_channel

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.inc = nn.Conv2d(in_channels, n_channel, kernel_size=3, padding=1, bias=True)

        self.res1 = ResBlock(n_channel, n_channel)
        self.down1 = nn.Conv2d(n_channel, n_channel * 2, 2, 2)
        self.mlp1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.afam1 = AFAM(64)

        self.res2 = ResBlock(n_channel * 2, n_channel * 2)
        self.down2 = nn.Conv2d(n_channel * 2, n_channel * 4, 2, 2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.afam2 = AFAM(128)

        self.res3 = ResBlock(n_channel * 4, n_channel * 4)
        self.down3 = nn.Conv2d(n_channel * 4, n_channel * 8, 2, 2)
        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 256)
        )
        self.afam3 = AFAM(256)

        self.res4 = ResBlock(n_channel * 8, n_channel * 8)
        self.hcb = HCB(n_channel * 8, n_channel * 2)
        self.offset4 = OffsetBlock(n_channel * 8, offset_channel, False)
        self.mrab4 = MRAB(n_channel * 8, n_channel * 8, offset_channel)

        self.up3 = UpSample(n_channel * 8, n_channel * 4)
        self.mlp4 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 256)
        )
        self.dconv3_1 = nn.Conv2d(n_channel * 8, n_channel * 4, 1, 1)
        self.offset3 = OffsetBlock(n_channel * 4, offset_channel, True)
        self.mrab3 = MRAB(n_channel * 4, n_channel * 4, offset_channel)

        self.up2 = UpSample(n_channel * 4, n_channel * 2)
        self.mlp5 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.dconv2_1 = nn.Conv2d(n_channel * 4, n_channel * 2, 1, 1)
        self.offset2 = OffsetBlock(n_channel * 2, offset_channel, True)
        self.mrab2 = MRAB(n_channel * 2, n_channel * 2, offset_channel)

        self.up1 = UpSample(n_channel * 2, n_channel)
        self.mlp6 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.dconv1_1 = nn.Conv2d(n_channel * 2, n_channel, 1, 1)
        self.offset1 = OffsetBlock(n_channel, offset_channel, True)
        self.mrab1 = MRAB(n_channel, n_channel, offset_channel)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.out = nn.Conv2d(n_channel, out_channels, 3, 1, 1)


    def forward(self, x, t, adjust):

        time_emb = self.time_mlp(t)

        x = self.inc(x)

        conv1 = self.res1(x)
        down1 = self.down1(conv1)
        condition1 = self.mlp1(time_emb)
        b, c = condition1.shape
        condition1 = rearrange(condition1, 'b c -> b c 1 1')
        down1 = down1 + condition1
        if adjust:
            down1 = self.afam1(down1)
        pool1 = self.lrelu(down1)

        conv2 = self.res2(pool1)
        down2 = self.down2(conv2)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')
        down2 = down2 + condition2
        if adjust:
            down2 = self.afam2(down2)
        pool2 = self.lrelu(down2)

        conv3 = self.res3(pool2)
        down3 = self.down3(conv3)
        condition3 = self.mlp3(time_emb)
        b, c = condition3.shape
        condition3 = rearrange(condition3, 'b c -> b c 1 1')
        down3 = down3 + condition3
        if adjust:
            down3 = self.afam3(down3)
        pool3 = self.lrelu(down3)

        conv4 = self.res4(pool3)
        conv4 = self.hcb(conv4)
        L4_offset = self.offset4(conv4, None)
        dconv4 = self.mrab4(conv4, L4_offset)

        up3 = self.up3(dconv4, conv3)
        condition4 = self.mlp4(time_emb)
        b, c = condition4.shape
        condition4 = rearrange(condition4, 'b c -> b c 1 1')
        up3 = up3 + condition4
        up3 = self.dconv3_1(up3)
        L3_offset = self.offset3(up3, L4_offset)
        dconv3 = self.mrab3(up3, L3_offset)

        up2 = self.up2(dconv3, conv2)
        condition5 = self.mlp5(time_emb)
        b, c = condition5.shape
        condition5 = rearrange(condition5, 'b c -> b c 1 1')
        up2 = up2 + condition5
        up2 = self.dconv2_1(up2)
        L2_offset = self.offset2(up2, L3_offset)
        dconv2 = self.mrab2(up2, L2_offset)

        up1 = self.up1(dconv2, conv1)
        condition6 = self.mlp6(time_emb)
        b, c = condition6.shape
        condition6 = rearrange(condition6, 'b c -> b c 1 1')
        up1 = up1 + condition6
        up1 = self.dconv1_1(up1)
        L1_offset = self.offset1(up1, L2_offset)
        dconv1 = self.mrab1(up1, L1_offset)

        out = self.out(dconv1)

        return out


class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Network, self).__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, t, y, x_end, adjust=True):
        x_middle = x[:, 1].unsqueeze(1)

        out = self.unet(x, t, adjust=adjust) + x_middle

        return out


# WeightNet of the one-shot learning framework
class WeightNet(nn.Module):
    def __init__(self, weight_num=10):
        super(WeightNet, self).__init__()
        init = torch.ones([1, weight_num, 1, 1]) / weight_num
        self.weights = nn.Parameter(init)

    def forward(self, x):
        weights = F.softmax(self.weights, 1)
        out = weights * x
        out = out.sum(dim=1, keepdim=True)

        return out, weights
