from fastai.vision.all import *
from fastai.vision import models
import torch
from typing import Tuple

def resnet50_7chan(pretrained=True):
    resnet = models.resnet50(pretrained=pretrained)
    conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        w = resnet.conv1.weight
        conv1.weight = nn.Parameter(torch.cat((w,
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:])
              ),dim=1))
    resnet.conv1 = conv1
    return resnet

def xresnet50_7chan(pretrained=True):
    xresnet = models.xresnet.xresnet50(pretrained=pretrained)
    conv1 = nn.Conv2d(7, 32, kernel_size=3, stride=2, padding=1, bias=False)
    if pretrained:
        w = xresnet[0][0].weight
        conv1.weight = nn.Parameter(torch.cat((w,
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:])
              ),dim=1))
    xresnet[0][0] = conv1
    return xresnet

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv2d_same(x,
                weight: torch.Tensor, 
                bias: Optional[torch.Tensor] = None, 
                stride: Tuple[int, int] = (1, 1),
                padding: Tuple[int, int] = (0, 0), 
                dilation: Tuple[int, int] = (1, 1), 
                groups: int = 1):
    """ Modified from rwightman's conv helper:
        https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/conv2d_layers.py
    """
    def _calc_same_pad(i: int, k: int, s: int, d: int):
        return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)
    
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return torch.nn.functional.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

def efficientnetb5_7chan(pretrained=True):
    effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b5_ns', pretrained=True)
    
    conv1 = Conv2dSame(7, 48, kernel_size=3, stride=2, bias=False)
    if pretrained:
        w = effnet.conv_stem.weight
        conv1.weight = nn.Parameter(torch.cat((w,
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:])
              ),dim=1))
    effnet.conv_stem = conv1
    return effnet

def resnest50_7chan(pretrained=True):
    resnest = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    conv1 = nn.Conv2d(7, 32, kernel_size=3, stride=2, padding=1, bias=False)
    if pretrained:
        w = resnest.conv1[0].weight
        conv1.weight = nn.Parameter(torch.cat((w,
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:]),
               0.5*(w[:,:1,:,:]+w[:,2:,:,:])
              ),dim=1))
    resnest.conv1[0] = conv1
    return resnest