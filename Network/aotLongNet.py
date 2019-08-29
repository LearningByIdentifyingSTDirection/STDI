import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from AOST.Network import backbone as Function

class aotLongNet(nn.Module):
    def __init__(self, param):
        super(aotLongNet, self).__init__()
        self.param = param
        self.videonet = getattr(Function, self.param.backbone)(sample_size=param.crop_size, sample_duration=16)
        ks = 3 if param.crop_size > 64 else 2
        self.fc_single = nn.Sequential(
                nn.Conv3d(512, 512, kernel_size=(1,ks,ks), stride=1, padding=0, bias=True),
                nn.ReLU(),
            )
        self.fc_cls = nn.Sequential(
                nn.Linear(512*4, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )

    def forward(self, vclip1, vclip2, vclip3, vclip4):
        vclips = [vclip1, vclip2, vclip3, vclip4]
        feats = []
        for vclip in vclips:
            feat = self.videonet.forward_as_STCubicbackbone(vclip)
            feat = self.fc_single(feat)
            feat = feat.view(-1, 512)
            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        score = self.fc_cls(feats)
        return score
