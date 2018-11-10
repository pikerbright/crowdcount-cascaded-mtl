#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from network import Conv2d, FC
import torchvision


class stack_pool(nn.Module):
    def __init__(self):
        super(stack_pool, self).__init__()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool2s1 = nn.MaxPool2d(2, stride=1)
        self.pool3s1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.padding = nn.ReplicationPad2d((0, 1, 0, 1))

    def forward(self, x):
        x1 = self.pool2(x)
        x2 = self.pool2s1(self.padding(x1))
        x3 = self.pool3s1(x2)
        y = (x1 + x2 + x3) / 3.0
        return y

vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512]

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    # pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # layers += [pool5, conv6,
    #            nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

class CMTL_VGG(nn.Module):

    def __init__(self, bn=False, num_classes=10):
        super(CMTL_VGG, self).__init__()
        self.num_classes = num_classes

        self.base = nn.Sequential(*vgg(vgg_cfg, 3))
        
        # base_model_dict = self.base.state_dict()
        # pretrained_model = getattr(torchvision.models, 'vgg16')(True)
        # pretrained_dict = pretrained_model.state_dict()
        # copy_dict = {}
        # for k, v in pretrained_dict.items():
        #     if 'features' in k:
        #         copy_dict[k[9:]] = v
        # copy_dict = {k: v for k, v in copy_dict.items() if k in base_model_dict}
        # base_model_dict.update(copy_dict)
        #
        # self.base.load_state_dict(base_model_dict)
        
        self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32, 32)),
                                        Conv2d(256, 4, 1, same_padding=True, NL='prelu', bn=bn))

        self.hl_prior_fc1 = FC(4 * 1024, 512, NL='prelu')
        self.hl_prior_fc2 = FC(512, 256, NL='prelu')
        self.hl_prior_fc3 = FC(256, self.num_classes, NL='prelu')

        self.de_stage = nn.Sequential(Conv2d(256, 128, 3, same_padding=True, NL='prelu', bn=bn),
                                        # nn.ReLU(inplace=True),
                                        Conv2d(128, 32, 3, same_padding=True, NL='prelu', bn=bn),
                                        #nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
                                        #nn.PReLU(),
                                        #nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True),
                                        #nn.PReLU(),
                                        # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0, bias=True),
                                        # nn.PReLU(),
                                        # nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                        Conv2d(32, 1, 1, same_padding=True, NL='relu', bn=bn))

    def forward(self, im_data):
        x = im_data
        for k in range(16): #conv3_3
            x = self.base[k](x)

        conv3_3 = x

        #for k in range(16, len(self.base)):
        #    x = self.base[k](x)

        #base_out = torch.cat([conv3_3, x], dim=1)
        base_out = x
        # print(base_out)
        
        x_hlp2 = self.hl_prior_2(base_out)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1)
        x_hlp = self.hl_prior_fc1(x_hlp2)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)

        x_den = base_out
        for m in self.de_stage:
            x_den = m(x_den)

        # print(x_den)
        return x_den, x_cls

class CMTL(nn.Module):
    '''
    Implementation of CNN-based Cascaded Multi-task Learning of High-level Prior and Density
    Estimation for Crowd Counting (Sindagi et al.)
    '''
    def __init__(self, bn=False, num_classes=10):
        super(CMTL, self).__init__()
        
        self.num_classes = num_classes        
        self.base_layer = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, NL='prelu', bn=bn),                                     
                                        Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn))
        
        self.hl_prior_1 = nn.Sequential(Conv2d( 32, 16, 9, same_padding=True, NL='prelu', bn=bn),
                                     stack_pool(),
                                     Conv2d(16, 32, 7, same_padding=True, NL='prelu', bn=bn),
                                     stack_pool(),
                                     Conv2d(32, 16, 7, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(16, 8,  7, same_padding=True, NL='prelu', bn=bn))
                
        self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32,32)),
                                        Conv2d( 8, 4, 1, same_padding=True, NL='prelu', bn=bn))
        
        self.hl_prior_fc1 = FC(4*1024,512, NL='prelu')
        self.hl_prior_fc2 = FC(512,256,    NL='prelu')
        self.hl_prior_fc3 = FC(256, self.num_classes,     NL='prelu')
        
        
        self.de_stage_1 = nn.Sequential(Conv2d( 32, 20, 7, same_padding=True, NL='prelu', bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, NL='prelu', bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, NL='prelu', bn=bn))
        
        self.de_stage_2 = nn.Sequential(Conv2d( 18, 24, 3, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 24, 32, 3, same_padding=True, NL='prelu', bn=bn),                                        
                                        nn.ConvTranspose2d(32,16,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16,8,4,stride=2,padding=1,output_padding=0,bias=True),
                                        nn.PReLU(),
                                        Conv2d(8, 1, 1, same_padding=True, NL='relu', bn=bn))
        
    def forward(self, im_data):
        x_base = self.base_layer(im_data)
        x_hlp1 = self.hl_prior_1(x_base)
        x_hlp2 = self.hl_prior_2(x_hlp1)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1) 
        x_hlp = self.hl_prior_fc1(x_hlp2)
#        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
#        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)        
        x_den = self.de_stage_1(x_base)        
        x_den = torch.cat((x_hlp1,x_den),1)
        x_den = self.de_stage_2(x_den)
        return x_den, x_cls

if __name__ == "__main__":
    img = np.ones((1, 3, 300, 300), dtype=np.float32)
    tensor = torch.from_numpy(img)
    inputs = Variable(tensor)
    net = CMTL_VGG()
    x_den, x_cls = net(inputs)
