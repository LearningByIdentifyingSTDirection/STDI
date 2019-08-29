
import torch
import torchvision
import numpy as np
import random
import os
import torch.utils.data.dataset as d
from PIL import Image
from AOST.Transforms import *
from AOST.Input.UnlabledDataset import UnlabledDataset

class aosLongDataset(d.Dataset):

	def __init__(self, istrain=True, param=None):
		self.L = 64
		self.param = param
		self.istrain = istrain

		subdir = ['Train',] if istrain else ['Eval',]
		if self.param.train_with_all_data:
			subdir = ['Train', 'Eval']
		self.backbone = UnlabledDataset([os.path.join(param.data_root_dir, subdir_item) for subdir_item in subdir], \
										self.L, self.param.data_dir_depth, backend='cv2')

	def __len__(self):
		return self.backbone.length()

	def __getitem__(self, index):
		vclips = self.backbone.getitem(index, continous=True)#[16x240x320x3] NOTE: the format is already RGB no need to convert

		#random crop [64x240x320x3] -> [64x80x80x3]
		if not self.param.no_resize:
			vclips = Resize(vclips, size=(171, 128))
		assert vclips.shape == (64, 128, 171, 3)

		vclips = RandomCrop(vclips, crop_size=(self.param.crop_size, self.param.crop_size))

		cls_id = random.randint(0,3)
		vclips = np.rot90(vclips, cls_id, axes=(1,2))
		vclips = np.ascontiguousarray(vclips)
		vclips = vclips.reshape(4,16,self.param.crop_size,self.param.crop_size,3)
		
		return {'vclip1':toTensor(vclips[0]), 'vclip2':toTensor(vclips[1]),  \
				'vclip3':toTensor(vclips[2]), 'vclip4':toTensor(vclips[3]), \
				'y':cls_id}

	# def vis(self, index):
	# 	info = self.__getitem__(index)
	# 	vclips = [info['vclip1'], info['vclip2'], info['vclip3'], info['vclip4'] ]# 4x[3x16x112x112]
	# 	vclips = torch.cat(vclips, dim=1)
	# 	vclips = vclips.transpose(0,1)
	# 	x = torchvision.utils.make_grid(vclips, nrow=16, normalize=True, scale_each=True)
	# 	ndarr = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
	# 	im = Image.fromarray(ndarr)

	# 	permutation = perm_table[info['y']%24]
	# 	im.save('../AVL_data/output/images/{}.png'.format(str(permutation)+'_'+str(index)))


