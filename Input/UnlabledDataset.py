import torch
import glob
import os
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

def dir_exploit(depth, root, leaf_nodes):
	if depth == 0:
		if len(os.listdir(root)) >= 64:
			leaf_nodes.append(root)
		return
	for sub_path in os.listdir(root):
		if os.path.isdir(os.path.join(root,sub_path)):
			dir_exploit(depth-1, os.path.join(root, sub_path), leaf_nodes)


class UnlabledDataset(object):
	def __init__(self, data_root_dir, last_frames, imgseq_depth=2, imgseq_format='%05d.jpg', backend='cv2'):
		self.data_root_dir = data_root_dir
		self.last_frames = last_frames
		self.imgseq_depth = imgseq_depth
		self.imgseq_format = imgseq_format
		self.backend = backend
		self.build_list()

	def build_list(self):
		self.imglist = []
		for subdir in self.data_root_dir:
			dir_exploit(self.imgseq_depth, subdir, self.imglist)
		random.shuffle(self.imglist)
		print(len(self.imglist))

	def fetch_video(self, dir, n_frames, sub_frames=16, continous=False):
		'''
			[...., ...., ...., ....]
			split n_frames into 4 parts.
			choose random sub_frames frames from each part
		'''
		n_imgs = len(glob.glob(os.path.join(dir, '*.jpg')))
		start = random.choice(range(n_imgs-n_frames+1))
		if self.backend == 'PIL':
			imgs = [Image.open(os.path.join(dir, self.imgseq_format%(i+1))) for i in range(start, start+n_frames)]
		else:
			if continous:
				imgs = [cv2.cvtColor(cv2.imread(os.path.join(dir, self.imgseq_format%(i+1))), cv2.COLOR_BGR2RGB) \
					for i in range(start, start+n_frames)]
				return np.array(imgs)

			indices = np.arange(start, start+n_frames).reshape(4, n_frames/4)
			tindices = []
			for i_temporal in range(4):
				start_point = random.randint(0,n_frames/4 - sub_frames)
				tindices.extend(list(indices[i_temporal][start_point:start_point+sub_frames]))

			imgs = [cv2.cvtColor(cv2.imread(os.path.join(dir, self.imgseq_format%(i+1))), cv2.COLOR_BGR2RGB) \
					for i in tindices]
			#[n_frames,[320,240,3]]
			imgs = np.array(imgs)
		return imgs

	def getitem(self, index, sub_frames=16, continous=False):
		return self.fetch_video(self.imglist[index], self.last_frames, sub_frames, continous)

	def length(self):
		return len(self.imglist)


if __name__ == '__main__':
	a = UnlabledDataset('../UCF_jpg/')
	a.build_list()
	print len(a.imglist)

