import torch
import numpy as np
from torch.autograd import Variable
import cv2

class GradCam:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			# outputs = self.model(*[I.float().cuda() for I in input])
			outputs = self.model(input[0].float().cuda(),input[1].float().cuda(),\
								 input[2].float().cuda(),input[3].float().cuda())
		else:
			outputs = self.model(*input)
			# features, output = self.extractor(input)
		if type(outputs) != list and type(outputs) != tuple:
			outputs = [outputs]
		origin_feature = self.model.last_conv_output


		cams = []
		for output in outputs:
			index = np.argmax(output.cpu().data.numpy())

			one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
			one_hot[0][index] = 1
			one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
			if self.cuda:
				one_hot = torch.sum(one_hot.cuda() * output)
			else:
				one_hot = torch.sum(one_hot * output)

			self.model.zero_grad()
			one_hot.backward(retain_graph=True)
			
			single_cams = []
			for i_feat in range(4):
				grads_val = self.model.last_conv_output_grad[i_feat].cpu().data.numpy()
				feature = origin_feature[i_feat].cpu().data.numpy()[0, :]

				weights = np.mean(grads_val, axis = (2, 3, 4))[0, :]
				cam = np.zeros(feature.shape[1 : ], dtype = np.float32)

				for i, w in enumerate(weights):
					cam += w * feature[i, :, :, :]

				cam = cam.squeeze()
				cam = np.maximum(cam, 0)
				cam = cv2.resize(cam, (171, 128))
				cam = cam - np.min(cam)
				cam = cam / np.max(cam)
				single_cams.append(cam)
			cams.append(single_cams)
		return cams

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))

