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
        
    @property
    def loss(self):
        return self.loss_mse + 0.0001*self.cross_entropy + self.rc_loss
    
    def forward(self,  im_data, gt_data=None, gt_cls_label=None, ce_weights=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                        
        density_map, density_cls_score = self.CCN(im_data)
        density_cls_prob = F.softmax(density_cls_score)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            gt_cls_label = network.np_to_variable(gt_cls_label, is_cuda=True, is_training=self.training,dtype=torch.FloatTensor)                        
            self.loss_mse, self.cross_entropy, self.rc_loss = self.build_loss(density_map, density_cls_prob, gt_data, gt_cls_label, ce_weights)
            
            
        return density_map
    
    def build_loss(self, density_map, density_cls_score, gt_data, gt_cls_label, ce_weights):
        loss_mse = self.loss_mse_fn(density_map, gt_data)        
        ce_weights = torch.Tensor(ce_weights)
        ce_weights = ce_weights.cuda()
        cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)

        density_map_size = density_map.size(1)*density_map.size(2)*density_map.size(3)
        Fy = torch.div(torch.sum(density_map.view(density_map.size(0), -1), dim = 1), density_map_size)
        Y = torch.div(torch.sum(gt_data.view(gt_data.size(0), -1), dim = 1), density_map_size)
        rc_loss = self.loss_mse_fn(Fy, Y)
        return loss_mse, cross_entropy, rc_loss

