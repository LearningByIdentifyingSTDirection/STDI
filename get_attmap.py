import torch
import numpy as np
import cv2
import os
import random
import glob
import argparse
from tqdm import tqdm
from AOST.Transforms import *
from AOST.tools.grad_cam_3d import *
import AOST.Network as Network

def get_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img/255.)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))

def vis_cam(model, frames_paths, output_paths):
    grad_cam = GradCam(model, True)
    for i_fp, frame_path in enumerate(tqdm(frames_paths)):
        n_imgs = len(glob.glob(os.path.join(frame_path, '*.jpg')))
        if n_imgs < 64:
            continue
        start = random.choice(range(n_imgs-64+1))
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(frame_path, '%05d.jpg'%(i+1))), cv2.COLOR_BGR2RGB) \
                for i in range(start, start+64)]
        imgs = np.array(imgs) # 64x240x320x3
        origin_imgs = imgs
        imgs = Resize(imgs, size=(80, 80))
        # imgs 64x112x112x3
        imgs = imgs.reshape(4,16,80,80,3)
        inputs = []
        inputs.append(toTensor(imgs[0]).unsqueeze(dim=0))
        inputs.append(toTensor(imgs[1]).unsqueeze(dim=0))
        inputs.append(toTensor(imgs[2]).unsqueeze(dim=0))
        inputs.append(toTensor(imgs[3]).unsqueeze(dim=0))

        cams = grad_cam(inputs)
        for i_cam, cam in enumerate(cams):
            for i_frame in range(64):
                origin_img = origin_imgs[i_frame]
                camed_img = get_cam_on_image(origin_img, cam[int(i_frame/16)])
                if not os.path.exists(output_paths[i_fp]):
                    os.makedirs(output_paths[i_fp])
                cv2.imwrite(os.path.join(output_paths[i_fp], '%05d_%d.jpg'%(i_frame+1, i_cam)), camed_img)
            os.system('ffmpeg -r 12 -i \'{}\' \'{}\''.format(\
                    os.path.join(output_paths[i_fp], '%05d_{}.jpg'.format(i_cam)),
                    os.path.join(output_paths[i_fp], '{}.avi'.format(i_cam))
                    )
                )
            os.system('rm {}'.format(os.path.join(output_paths[i_fp].replace(' ','\\ '), '*_%d.jpg'%(i_cam))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--val_path', default='', type=str)
    parser.add_argument('--sav_path', default='', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.sav_path):
        os.makedirs(args.sav_path)

    frames_paths = []
    output_paths = []
    for label in os.listdir(args.val_path):
        lbpath = os.path.join(args.val_path, label)
        for vid in os.listdir(lbpath):
            vidpath = os.path.join(lbpath, vid)
            frames_paths.append(vidpath)
            op = os.path.join(args.sav_path, label, vid)
            # if not os.path.exists(op):
            #     os.makedirs(op)
            output_paths.append(op)

    class param:
        pass
    para = param()
    para.backbone = 'resnet18'
    para.crop_size = 80
    network = Network.aostLongNet(para)
    state_dict = torch.load(args.resume_path)
    state_dict = {key.replace('module.',''):value for key, value in state_dict.items()}
    network.load_state_dict(state_dict)

    vis_cam(network, frames_paths, output_paths)





