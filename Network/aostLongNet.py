import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from AOST.Network import backbone as Function

class aostLongNet(nn.Module):
    def __init__(self, param):
        super(aostLongNet, self).__init__()
        self.param = param
        self.videonet = getattr(Function, self.param.backbone)(sample_size=param.crop_size, sample_duration=16)
        ks = 3 if param.crop_size > 64 else 2
        self.sfc_single = nn.Sequential(
                nn.Conv3d(512, 512, kernel_size=(1,ks,ks), stride=1, padding=0, bias=True),
                nn.ReLU(),
            )
        self.sfc_cls = nn.Sequential(
                nn.Linear(512*4, 512),
                nn.ReLU(),
                nn.Linear(512, 4)
            )

        self.tfc_single = nn.Sequential(
                nn.Conv3d(512, 512, kernel_size=(1,ks,ks), stride=1, padding=0, bias=True),
                nn.ReLU(),
            )
        self.tfc_cls = nn.Sequential(
                nn.Linear(512*4, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )

    def save_gradient(self, grad, index):
        self.last_conv_output_grad[index] = grad

    def forward(self, vclip1, vclip2, vclip3, vclip4):
        vclips = [vclip1, vclip2, vclip3, vclip4]
        sfeats = []
        tfeats = []
        self.last_conv_output = [0,0,0,0]
        self.last_conv_output_grad = [0,0,0,0]
        
        for i_vclip, vclip in enumerate(vclips):
            feat = self.videonet.forward_as_STCubicbackbone(vclip)

            self.last_conv_output[i_vclip] = feat
            self.last_conv_output[i_vclip].register_hook(partial(self.save_gradient, index=i_vclip))

            sfeat = self.sfc_single(feat)
            sfeat = sfeat.view(-1, 512)
            sfeats.append(sfeat)

            tfeat = self.tfc_single(feat)
            tfeat = tfeat.view(-1, 512)
            tfeats.append(tfeat)            

        sfeats = torch.cat(sfeats, dim=1)
        tfeats = torch.cat(tfeats, dim=1)
        sscore = self.sfc_cls(sfeats)
        tscore = self.tfc_cls(tfeats)
        return sscore, tscore
