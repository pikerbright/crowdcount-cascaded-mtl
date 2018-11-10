import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.crowd_count import CrowdCounter
from src import network
import cv2

model_path = './final_models/cmtl_shtechB_39200.h5'

def read_gray_img(img_path):
    bgr = cv2.imread(img_path)
    #bgr = cv2.resize(bgr, (225, 225))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    plt.imshow(gray_3)
    plt.show()

    img = gray_3 / 255.
    img = (img - [0.5, 0.5, 0.5]) / [0.229, 0.229, 0.229]
    img = np.transpose(img, [2, 0, 1])
    img = img.reshape((1, 3, img.shape[1], img.shape[2]))

    # gray = gray.reshape((1, 1, gray.shape[0], gray.shape[1]))
    return img

def demo(img_path):
    net = CrowdCounter()
    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.eval()

    # net.load_state_dict(torch.load('checkpoint/crowd_net19.pth', map_location='cpu'))
    input_img = read_gray_img(img_path)
    # input_img = torch.autograd.Variable(torch.from_numpy(input_img))
    print(input_img.shape)
    # input_image = input_image.view(1, 3, 255, 255)
    heat_map = net(input_img)
    print(heat_map.size())
    heat_map = torch.squeeze(heat_map)
    heat_map = heat_map.data.numpy()
    print(np.sum(heat_map))
    plt.imshow(heat_map, cmap='hot')
    plt.savefig('test.jpg')
    plt.show()


if __name__ == '__main__':
    # demo('demo/demo4.jpg')
    demo('data/formatted_trainval/shanghaitech_part_B_patches_9/train/1_1.jpg')
    # demo('/Users/zhuliang/work/huangpu/00006.jpg')

