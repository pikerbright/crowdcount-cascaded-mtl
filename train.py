import os
import torch
import numpy as np
import sys
from datetime import datetime
import argparse

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--opt', '--optimizer', default="Adam",
                    metavar='optimizer', help='optimizer type')

args = parser.parse_args()
print "LR: ", args.lr, "weight decay", args.weight_decay, "momentum", args.momentum, "optimizer", args.opt
method = 'cmtl' #method name - used for saving model file
dataset_name = 'shtechB' #dataset name - used for saving model file
output_dir = './saved_models/' #model files are saved here

#train and validation paths
train_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/train1'
train_gt_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/train1'
val_gt_path = './data/formatted_trainval/shanghaitech_part_B_patches_9/train_den'

#training configuration
start_step = 0
end_step = 40000
disp_interval = 500
log_interval = 250

#Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoardO
exp_name = None # the previous experiment name in TensorBoard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rand_seed = None#64678
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rand_seed)
    
#loadt training and validation data
data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
class_wts = data_loader.get_classifier_weights()
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

resume = False
#load net and initialize it
net = CrowdCounter(ce_weights=class_wts)

if resume:
    resume_model = "./model_bak/cmtl_shtechA_204.h5"
    network.load_net(resume_model, net)
    print("Resume ", resume_model)
else:
    # network.weights_normal_init(net, dev=0.01)
    net.init_weight()

net = net.to(device)
net.train()

policies = net.get_optim_policies()
for group in policies:
    print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
        group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
if args.opt == "Adam":
    optimizer = torch.optim.Adam(policies, lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, epoch, lr_steps):
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

best_mae = sys.maxint

iteration = 0
adjust_learning_rate(optimizer, 0, (1, 2))
for epoch in range(start_step, end_step+1):    
    step = -1
    train_loss = 0
#    adjust_learning_rate(optimizer, epoch)
    for blob in data_loader:
        # adjust_learning_rate(optimizer, epoch, (200, 400, 600))

        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        gt_class_label = blob['gt_class_label']       

        #data augmentation on the fly
        if np.random.uniform() > 0.5:
            #randomly flip input image and density
            im_data = np.flip(im_data,3).copy()
            gt_data = np.flip(gt_data,3).copy()
        if np.random.uniform() > 0.5:
            #add random noise to the input image
            im_data = im_data + np.random.uniform(-0.1,0.1,size=im_data.shape)

        l2_reg = None
        for name, W in net.named_parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        density_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if True:#step % disp_interval == 0:            
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)    
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            utils.save_results(im_data,gt_data,density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.5f, et_cnt: %4.5f' % (epoch,
                step, 1./fps, gt_count,et_count)
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True    
    
       
        if re_cnt:                                
            t.tic()
            re_cnt = False

        iteration += 1

    if (epoch % 1400 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, net) 
        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)

    print(train_loss)

