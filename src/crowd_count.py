import torch
import torch.nn as nn
import torch.nn.functional as F

import network
from models import CMTL, CMTL_VGG

class RCLoss(nn.Module):

    def __init__(self):
        super(RCLoss, self).__init__()

    def forward(self, input, target):
        loss_mse_fn = nn.MSELoss()
        loss = loss_mse_fn(input, target)
        loss = torch.div(loss, torch.pow(target, 2) + 1)

        return loss

class CrowdCounter(nn.Module):
    def __init__(self, ce_weights=None):
        super(CrowdCounter, self).__init__()        
        self.CCN = CMTL_VGG()
        self.loss_mse_fn = nn.MSELoss()
        self.rc_loss_fn = RCLoss()

    def init_weight(self):
        network.weights_normal_init(self.CCN.base, dev=0.01)
        network.weights_normal_init(self.CCN.conv_concat1_2x, dev=0.01)
        network.weights_normal_init(self.CCN.p_conv, dev=0.01)

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        head_weight = []
        head_bias = []
        deconv_weight = []

        for m in self.CCN.base.modules():
            if isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                base_bias.append(ps[1])

        head_modules = list(self.CCN.p_conv.modules())

        for m in head_modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                ps = list(m.parameters())
                head_weight.append(ps[0])
                head_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                head_weight.append(ps[0])
                if len(ps) == 2:
                    head_bias.append(ps[1])
            elif isinstance(m, torch.nn.PReLU):
                ps = list(m.parameters())
                head_weight.append(ps[0])

        for m in self.CCN.conv_concat1_2x.modules():
            ps = list(m.parameters())
            deconv_weight.append(ps[0])

        return [
            {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "base_weight"},
            {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "base_bias"},
            {'params': head_weight, 'lr_mult': 1, 'decay_mult': 0, 'name': "head_weight"},
            {'params': head_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "head_bias"},
            {'params': deconv_weight, 'lr_mult': 0, 'decay_mult': 0, 'name': "deconv_weight"}
        ]

    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,  im_data, gt_data=None, gt_cls_label=None, ce_weights=None):        
        im_data = network.np_to_variable(im_data, is_training=self.training)
        density_map = self.CCN(im_data)

        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_training=self.training)
            self.loss_mse = self.build_loss(density_map, gt_data)
            
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)        

        return loss_mse

