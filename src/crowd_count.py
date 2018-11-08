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
        if ce_weights is not None:
            ce_weights = torch.Tensor(ce_weights)
            ce_weights = ce_weights.cuda()
        self.loss_mse_fn = nn.MSELoss()
        self.loss_bce_fn = nn.BCELoss(weight=ce_weights)
        self.rc_loss_fn = RCLoss()

    def init_weight(self):
#        network.weights_normal_init(self.CCN.base, dev=0.01)
        network.weights_normal_init(self.CCN.hl_prior_2, dev=0.01)
        network.weights_normal_init(self.CCN.hl_prior_fc1, dev=0.01)
        network.weights_normal_init(self.CCN.hl_prior_fc2, dev=0.01)
        network.weights_normal_init(self.CCN.hl_prior_fc3, dev=0.01)
        network.weights_normal_init(self.CCN.de_stage, dev=0.01)

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        head_weight = []
        head_bias = []

        for m in self.CCN.base.modules():
            if isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                base_weight.append(ps[0])
                base_bias.append(ps[1])

        head_modules = list(self.CCN.hl_prior_2.modules()) + list(self.CCN.hl_prior_fc1.modules()) \
                        + list(self.CCN.hl_prior_fc2.modules()) + list(self.CCN.hl_prior_fc3.modules()) \
                        + list(self.CCN.de_stage.modules())

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

        return [
            {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "base_weight"},
            {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "base_bias"},
            {'params': head_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "head_weight"},
            {'params': head_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "head_bias"}
        ]

    @property
    def loss(self):
        return self.loss_mse# + 0.0001*self.cross_entropy
    
    def forward(self,  im_data, gt_data=None, gt_cls_label=None, ce_weights=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                        
        density_map, density_cls_score = self.CCN(im_data)
        density_cls_prob = F.softmax(density_cls_score)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            gt_cls_label = network.np_to_variable(gt_cls_label, is_cuda=True, is_training=self.training,dtype=torch.FloatTensor)                        
            self.loss_mse, self.cross_entropy = self.build_loss(density_map, density_cls_prob, gt_data, gt_cls_label, ce_weights)
            
            
        return density_map
    
    def build_loss(self, density_map, density_cls_score, gt_data, gt_cls_label, ce_weights):
        loss_mse = self.loss_mse_fn(density_map, gt_data)        
        ce_weights = torch.Tensor(ce_weights)
        ce_weights = ce_weights.cuda()
        cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)

        return loss_mse, cross_entropy

