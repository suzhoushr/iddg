import pdb

import torch.nn as nn
import torch

class Conv2DReLU(nn.Module):
    def __init__(self, in_chan, out_chan, n_emb=None, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True, norm_type='instance'):
        super(Conv2DReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        if norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_chan)
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(out_chan)
        else:
            self.norm_layer = None
        self.relu = nn.ReLU(inplace=True)

        self.emb_layers = None
        if n_emb is not None:
            self.emb_layers = self.emb_layers = nn.Linear(n_emb, 2 * out_chan)

    def forward(self, x, emb=None):
        feat = self.conv(x)
        if self.norm_layer is not None:
            feat = self.norm_layer(feat)
        if emb is not None:
            assert self.emb_layers is not None, 'you forget to set the emb_layers'
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(feat.shape):
                emb_out = emb_out[..., None]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            feat = feat * (1 + scale) + shift
        feat = self.relu(feat)
        return feat

class TConv2DReLU(nn.Module):
    def __init__(self, in_chan, out_chan, n_emb=None, ks=3, stride=2, padding=1,
                 dilation=1, groups=1, bias=True, norm_type='instance'):
        super(TConv2DReLU, self).__init__()
        output_padding = padding
        if ks == 4:
            output_padding = padding - 1 # 0
        self.conv = nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=ks, stride=stride,
                               padding=padding, groups=groups, dilation=dilation, output_padding=output_padding, bias=bias)
        if norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_chan)
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(out_chan)
        else:
            self.norm_layer = None
        self.relu = nn.ReLU(inplace=True)

        self.emb_layers = None
        if n_emb is not None:
            self.emb_layers = nn.Linear(n_emb, 2 * out_chan)

    def forward(self, x, emb=None):
        feat = self.conv(x)
        if self.norm_layer is not None:
            feat = self.norm_layer(feat)
        if emb is not None:
            assert self.emb_layers is not None, 'you forget to set the emb_layers'
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(feat.shape):
                emb_out = emb_out[..., None]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            feat = feat * (1 + scale) + shift
        feat = self.relu(feat)
        return feat
    
class ResBlockConv2DReLU(nn.Module):
    def __init__(self, in_chan, out_chan, n_emb=None, ks=3, stride=1,
                 padding=1, dilation=1, bias=True, norm_type='instance'):
        super(ResBlockConv2DReLU, self).__init__()

        self.use_res_connect = stride == 1 and in_chan == out_chan

        self.res_block = nn.Sequential(
            Conv2DReLU(in_chan=in_chan, out_chan=in_chan, 
                       ks=ks, stride=stride, 
                       padding=padding, dilation=dilation, 
                       groups=1, norm_type=norm_type),
            nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=1, bias=bias),)

        if norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_chan)
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d(out_chan)
        else:
            self.norm_layer = None  

        self.emb_layers = None
        if n_emb is not None:
            self.emb_layers = self.emb_layers = nn.Linear(n_emb, 2 * out_chan)

    def forward(self, x, emb=None):
        feat = self.res_block(x)
        if self.norm_layer is not None:
            feat = self.norm_layer(feat)

        if emb is not None:
            assert self.emb_layers is not None, 'you forget to set the emb_layers'
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(feat.shape):
                emb_out = emb_out[..., None]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            feat = feat * (1 + scale) + shift

        if self.use_res_connect:
            return x + feat
        else:
            return feat
        
class Pix2PixSimple(nn.Module):
    def __init__(self, input_nc=3,
                 output_nc=3,
                 ngf=8,
                 n_emb=512,
                 n_classes=1000,
                 n_downsampling=3,
                 n_blocks=12,
                 ratio_decode=1.5,
                 norm_type='instance'):
        super(Pix2PixSimple, self).__init__()

        self.emb_layer = nn.Embedding(n_classes, n_emb)

        self.input_layers = Conv2DReLU(input_nc, ngf, ks=3, stride=2, padding=1, dilation=1, groups=1, norm_type=norm_type)

        ### downsample
        model = list()
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [Conv2DReLU(ngf * mult, ngf * mult * 2, n_emb=n_emb, ks=3, stride=2, padding=1, dilation=1, groups=1, norm_type=norm_type)]
        model += [Conv2DReLU(ngf * mult * 2, ngf * mult * 2, n_emb=n_emb, ks=3, stride=1, padding=1, dilation=1, groups=1, norm_type=norm_type)]

        ### resnet blocks
        mult = 2 ** n_downsampling
        out_chan = ngf * mult
        for i in range(n_blocks):
            model += [ResBlockConv2DReLU(out_chan, out_chan, n_emb=n_emb, ks=3, stride=1, padding=1, dilation=1, bias=True, norm_type=norm_type)]
        self.model = nn.Sequential(*model)

        in_chan = out_chan
        out_chan = int(in_chan / ratio_decode)
        out_chan = out_chan - out_chan % 4
        self.up_x2 = TConv2DReLU(in_chan=in_chan, out_chan=out_chan, n_emb=n_emb, ks=3, stride=2, padding=1, dilation=1,
                                       groups=1, norm_type=norm_type)

        in_chan = out_chan
        out_chan = int(in_chan / ratio_decode)
        out_chan = out_chan - out_chan % 4
        self.up_x4 = TConv2DReLU(in_chan=in_chan, out_chan=out_chan, n_emb=n_emb, ks=3, stride=2, padding=1, dilation=1,
                                       groups=1, norm_type=norm_type)

        in_chan = out_chan
        out_chan = int(in_chan / ratio_decode)
        out_chan = out_chan - out_chan % 4
        self.up_x8 = TConv2DReLU(in_chan=in_chan, out_chan=out_chan, n_emb=n_emb, ks=3, stride=2, padding=1, dilation=1,
                                       groups=1, norm_type=norm_type)

        in_chan = out_chan
        out_chan = int(in_chan / ratio_decode)
        out_chan = out_chan - out_chan % 4
        self.up_x16 = TConv2DReLU(in_chan=in_chan, out_chan=out_chan, n_emb=n_emb, ks=3, stride=2, padding=1, dilation=1,
                                       groups=1, norm_type=norm_type)

        in_chan = out_chan
        out_chan = output_nc
        # self.out = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
        #                          nn.Tanh(),)
        self.out = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y=None):
        emb = None
        if y is not None:
            emb = self.emb_layer(y)
        feats = self.input_layers(x)
        for model in self.model:
            feats = model(feats, emb)
        # pdb.set_trace()
        feats = self.up_x2(feats, emb)
        feats = self.up_x4(feats, emb)
        feats = self.up_x8(feats, emb)
        feats = self.up_x16(feats, emb)
        out = self.out(feats)
        # pdb.set_trace()
        return out
    
