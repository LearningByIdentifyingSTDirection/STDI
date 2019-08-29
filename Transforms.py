import torch
import random
import cv2
import numpy as np

def toTensor(video):
    # [TxHxWxC] -> [CxTxHxW] -> div 255
    video = torch.from_numpy(video.transpose((3,0,1,2)))
    if isinstance(video, torch.ByteTensor):
        return video.float().div(255)
    else:
        return video

def RandomCrop(video, crop_size):
    # [TxHxWxC] -> [TxtHxtWxC]
    # video should be [TxHxWxC]
    # crop_size should be [HxW]
    h, w = video.shape[1:3]
    th, tw = crop_size
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    target_video = video[:,i:i+th,j:j+tw,:]
    return target_video

def Resize(video, size):
    target = []
    for i in range(video.shape[0]):
        target.append(cv2.resize(video[i], size))
    return np.array(target)