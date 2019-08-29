import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import os
import glob
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def vis_2Dfilter(pth, key):
    parameters = torch.load(pth)
    tensor = parameters[key]
    # tensor [outsize, insize, kernel_size(w), kernelsize(h)]
    sz = tensor.size()
    n_filters = sz[0]
    n_in = sz[1]
    W, H = sz[2], sz[3]
    assert n_in == 3

    imgs = []
    for i_filter in range(n_filters):
        img = tensor[i_filter,...]
        img = F.interpolate(img.unsqueeze(0), [56,56], mode='bilinear').squeeze(0)
        imgs.append(img)
    x = torchvision.utils.make_grid(imgs, nrow=7, normalize=True, scale_each=True)
    ndarr = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save('../AVL_data/output/vis/vis.png')

def vis_3Dfilter2(pth, key, style='time_major'):
    parameters = torch.load(pth)
    tensor = parameters[key]
    # tensor [outsize, insize, kernel_size(t), kernel_size(w), kernelsize(h)]
    sz = tensor.size()
    n_filters = sz[0]
    n_in = sz[1]
    T, W, H = sz[2], sz[3], sz[4]
    assert n_in == 3
    tensor = tensor.transpose(1,2)
    # tensor [outsize, kernel_size(t), insize, kernel_size(w), kernelsize(h)]

    imgs = []
    for i_filter in range(n_filters):
        for t in range(T):
            img = tensor[i_filter,t,...]
            img = F.interpolate(img.unsqueeze(0), [56,56], mode='bilinear').squeeze(0)
            img = (img - img.min())/img.max()
            imgs.append(img)
    x = torchvision.utils.make_grid(imgs, nrow=7, normalize=True, scale_each=True)
    if style == 'time_major':
        x = []
        timgs = []
        for t in range(7):
            for f in range(64):
                timgs.append(imgs[f*7+t])
        imgs = timgs
        for t in range(7):
            x.append(torchvision.utils.make_grid(imgs[t*64:(t+1)*64], nrow=8, normalize=True, scale_each=True))
    return x

def vis_3Dfilter(pth, key, style='filter_major'):
    parameters = torch.load(pth)
    tensor = parameters[key]
    # tensor [outsize, insize, kernel_size(t), kernel_size(w), kernelsize(h)]
    sz = tensor.size()
    n_filters = sz[0]
    n_in = sz[1]
    T, W, H = sz[2], sz[3], sz[4]
    assert n_in == 3
    tensor = tensor.transpose(1,2)
    # tensor [outsize, kernel_size(t), insize, kernel_size(w), kernelsize(h)]

    imgs = []
    for i_filter in range(n_filters):
        for t in range(T):
            img = tensor[i_filter,t,...]
            img = F.interpolate(img.unsqueeze(0), [56,56], mode='bilinear').squeeze(0)
            img = (img - img.min())/img.max()
            imgs.append(img)
    x = torchvision.utils.make_grid(imgs, nrow=7, normalize=True, scale_each=True)
    if style == 'time_major':
        timgs = []
        for f in range(64):
            for t in range(7):
                timgs.append(imgs[7*t + f])
        imgs = timgs
        x = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, scale_each=True)
    return x

def save(x, path):
    ndarr = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.save('../AVL_data/output/vis/vis.png')
    im.save(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='aost80', type=str)
    parser.add_argument('--dir', default='none', type=str)
    parser.add_argument('--target_dir', default='none', type=str)
    parser.add_argument('--begin', default=0, type=int)
    args = parser.parse_args()
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment='{}.conv1'.format(args.task))
    pths = glob.glob(os.path.join(args.dir, 'save_*.pth'))
    pths = sorted(pths, key=lambda x:int(os.path.basename(x)[5:-4]))
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    for i_pth, pth in enumerate(pths):
        assert int(os.path.basename(pth)[5:-4]) == (i_pth * 10)
        print('PROCESSING {}'.format(pth))
        x = vis_3Dfilter(pth, 'module.videonet.conv1.weight', 'filter_major')
        save(x, os.path.join(args.target_dir, '%05d.png'%(i_pth+args.begin/10)))
        writer.add_image('{}'.format(args.task), x, (i_pth)*5+args.begin)

    #vis_2Dfilter('../AVL_data/ac_model/I3d_partialperm_transfer.pth','module.Conv3d_1a_7x7.conv3d.weight')
    #vis_3Dfilter('../AVL_data/R2model/permutation.pth','module.videonet.conv1.weight', 'time_major')
