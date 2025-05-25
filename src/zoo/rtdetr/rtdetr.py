

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np


from src.core import register
from thop import profile

__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale # multi_scale: [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        
    def forward(self, x, targets=None): # 假设x =[2,3,640,640]经过图像与处理大小都是640*640
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale) # 随机抽取到的尺寸是576，然后就会rize为那个随机抽取到的尺寸
            x = F.interpolate(x, size=[sz, sz])  # 通过双线性插值的方法就会resize为x=[2,3,576,576]
        # 依次进入backbone encoder和decoder模块

        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
