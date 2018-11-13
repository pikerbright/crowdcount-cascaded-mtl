from crowd_count import CrowdCounter
import network
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(trained_model, data_loader):    
    net = CrowdCounter()
    network.load_net(trained_model, net)
    net = net.to(device)
    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:                        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        print "gt_count", gt_count,"et_count", et_count
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))        
    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae,mse
