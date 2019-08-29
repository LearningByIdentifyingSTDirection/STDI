from __future__ import division
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import torch.utils.data.dataloader as dl
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import AOST.Input as Input
import AOST.Network as Network
import os
import argparse
import glob
import time
from tensorboardX import SummaryWriter

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_with_all_data', action='store_true')
    parser.set_defaults(train_with_all_data=False)
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--gpus', default='0,1,2,3', type=str)
    parser.add_argument('--data_root_dir', default='', type=str)
    parser.add_argument('--data_dir_depth', default=2, type=int)
    parser.add_argument('--task', type=str)
    parser.add_argument('--experiment_folder', default='./experiments/', type=str)
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--no_resize', action='store_true')
    parser.set_defaults(no_resize=True)
    parser.add_argument('--resume_path', default='', type=str)
    parser.add_argument('--crop_size', default=80, type=int)
    args = parser.parse_args()
    args.gpus = map(lambda x:int(x), args.gpus.split(','))
    return args

def get_network(param):
    if param.task == 'aos':
        return Network.aosLongNet(param)
    elif param.task == 'aot':
        return Network.aotLongNet(param)
    elif param.task == 'aost':
        return Network.aostLongNet(param)

def get_input(param, istrain):
    if param.task == 'aos':
        return Input.aosLongDataset(istrain, param)
    elif param.task == 'aot':
        return Input.aotLongDataset(istrain, param)
    elif param.task == 'aost':
        return Input.aostLongDataset(istrain, param)

def get_experiment(param):
    folder = param.experiment_folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_latest_checkpoint(folder):
    checkpoints = glob.glob(os.path.join(folder, '*.pth'))
    checkpoints = sorted(checkpoints, key=lambda x:int(os.path.basename(x)[5:-4]))
    return checkpoints[-1]


class runner(object):
    def __init__(self, param):
        cudnn.benchmark = True
        self.param = parse()
        self.net = get_network(param)
        self.device = torch.device("cuda:{}".format(self.param.gpus[0]))
        self.net = nn.DataParallel(self.net, device_ids=self.param.gpus)
        self.net.to(self.device)
        self.experiment_folder = get_experiment(self.param)


    def eval(self):
        with torch.no_grad():
            self.net.eval()
            self.epoch_accuracy = utils.recorder.AccuracyRecorder()
            if self.param.task == 'aost':
                self.epoch_saccuracy = utils.recorder.AccuracyRecorder()
                self.epoch_taccuracy = utils.recorder.AccuracyRecorder()
            self.epoch_loss = utils.recorder.AverageRecorder()
            if self.param.task == 'aost':
                self.epoch_sloss = utils.recorder.AverageRecorder()
                self.epoch_tloss = utils.recorder.AverageRecorder()
            self.data_time = utils.recorder.AverageRecorder()
            self.gpu_time = utils.recorder.AverageRecorder()
            self.testset = get_input(self.param, False)
            self.eval_dataloader = dl.DataLoader(self.testset, batch_size=self.param.eval_batch_size, shuffle=True, num_workers=4)
            self.net.load_state_dict(torch.load(self.param.resume_path))
            #try: self.net.load_state_dict(torch.load(get_latest_checkpoint(self.experiment_folder)))
            #except: print('No Such BreakPoint')
            self.epoch_loop(self.eval_dataloader, istrain=False)

    def train(self, load=True):
        self.net.train()
        self.trainset = get_input(self.param, True)
        self.train_dataloader = dl.DataLoader(self.trainset, batch_size=self.param.batch_size, shuffle=True, num_workers=16)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.param.lr, momentum=0.9, weight_decay=0.)
        self.max_epoch = 2000
        self.epoch_loss = utils.recorder.AverageRecorder()
        self.loss_recorder = utils.recorder.FileRecorder(os.path.join(self.experiment_folder, 'loss.log'))
        self.writer = SummaryWriter(comment=self.param.task)
        self.data_time = utils.recorder.AverageRecorder()
        self.gpu_time = utils.recorder.AverageRecorder()
        if self.param.task == 'aost':
            self.epoch_sloss = utils.recorder.AverageRecorder()
            self.epoch_tloss = utils.recorder.AverageRecorder()
        if self.param.resume_path != '' and self.param.resume_path != 'none':
            self.net.load_state_dict(torch.load(self.param.resume_path))
        #if load:
            #try: self.net.load_state_dict(torch.load(get_latest_checkpoint(self.experiment_folder)))
            #except: print('No Such BreakPoint')
        for i_epoch in range(self.max_epoch):
            print('Epoch:{}'.format(i_epoch))
            self.epoch_loop(self.train_dataloader)
            if self.param.task == 'aos' or self.param.task == 'aot':
                self.loss_recorder.add([i_epoch, self.epoch_loss.get()])
                self.writer.add_scalar('loss', self.epoch_loss.get(), i_epoch)
            else:
                self.loss_recorder.add([i_epoch, self.epoch_loss.get(), \
                                        self.epoch_sloss.get(), self.epoch_tloss.get()])
                self.writer.add_scalar('loss', self.epoch_loss.get(), i_epoch)
                self.writer.add_scalar('sloss', self.epoch_sloss.get(), i_epoch)
                self.writer.add_scalar('tloss', self.epoch_tloss.get(), i_epoch)
            save_name = 'save_{}.pth'.format(int(i_epoch/10)*10)
            torch.save(self.net.state_dict(), os.path.join(self.experiment_folder, save_name))


    def epoch_loop(self, dataloader, istrain=True):
        if istrain:
            self.epoch_loss.clear()
            self.data_time.clear()
            self.gpu_time.clear()
            if self.param.task == 'aost':
                self.epoch_sloss.clear()
                self.epoch_tloss.clear()
        else:
            self.epoch_accuracy.clear()
            if self.param.task == 'aost':
                self.epoch_saccuracy.clear()
                self.epoch_taccuracy.clear()

        pbar = tqdm(dataloader)
        criterion = nn.CrossEntropyLoss().to(self.device)
        zero_time = time.time()
        for i_batch, data in enumerate(pbar):
            # fetch
            vclip1 = data['vclip1'].float().to(self.device)
            vclip2 = data['vclip2'].float().to(self.device)
            vclip3 = data['vclip3'].float().to(self.device)
            vclip4 = data['vclip4'].float().to(self.device)
            one_time = time.time()
            self.data_time.add(one_time-zero_time)
            if istrain: 
                self.optimizer.zero_grad()

            if self.param.task == 'aos' or self.param.task == 'aot':
                y = data['y'].long().to(self.device)
                logits = self.net(vclip1, vclip2, vclip3, vclip4)
                logits = logits.to(self.device)
                loss = criterion(logits, y)
            else:
                sy = data['sy'].long().to(self.device)
                ty = data['ty'].long().to(self.device)
                slogits, tlogits = self.net(vclip1, vclip2, vclip3, vclip4)
                slogits = slogits.to(self.device)
                tlogits = tlogits.to(self.device)
                sloss = criterion(slogits, sy)
                tloss = criterion(tlogits, ty)
                self.epoch_sloss.add(sloss.item())
                self.epoch_tloss.add(tloss.item())
                loss = sloss + tloss

            self.epoch_loss.add(loss.item())

            if istrain:
                loss.backward()
                self.optimizer.step()
            else:
                if self.param.task == 'aos' or self.param.task == 'aot':
                    judge = logits.argmax(dim=1)
                    correct = (judge == y).sum().item()
                    total = y.size()[0]
                    self.epoch_accuracy.add(correct, total)
                else:
                    sjudge = slogits.argmax(dim=1)
                    scorrect = (sjudge == sy).sum().item()
                    tjudge = tlogits.argmax(dim=1)
                    tcorrect = (tjudge == ty).sum().item()
                    total = sy.size()[0]
                    self.epoch_saccuracy.add(scorrect, total)
                    self.epoch_taccuracy.add(tcorrect, total)
            zero_time = time.time()
            self.gpu_time.add(zero_time-one_time)

            if istrain:
                if self.param.task == 'aos' or self.param.task == 'aot':
                    pbar.set_postfix(info='loss:{:.4f}/{:.4f}, dgtime:{:.4f}/{:.4f}'\
                            .format(loss.item(), self.epoch_loss.get(), self.data_time.get(), self.gpu_time.get()))
                else:
                    pbar.set_postfix(info='sloss:{:.4f}/{:.4f}, tloss:{:.4f}/{:.4f}, total_loss:{:.4f}/{:.4f}, dgtime:{:.4f}/{:.4f}'\
                            .format(sloss.item(), self.epoch_sloss.get(), tloss.item(), self.epoch_tloss.get(), loss.item(), self.epoch_loss.get(),\
                                    self.data_time.get(), self.gpu_time.get()))
            else:
                if self.param.task == 'aos' or self.param.task == 'aot':
                    pbar.set_postfix(info='acc:{:.4f}'\
                                    .format(self.epoch_accuracy.get()))
                else:
                    pbar.set_postfix(info='sacc:{:.4f}, tacc:{:.4f}'\
                                    .format(self.epoch_saccuracy.get(), self.epoch_taccuracy.get()))


if __name__ == '__main__':
    param = parse()
    print param
    handle = runner(param)
    if param.stage == 'train':
        handle.train(load=False)
    else:
        handle.eval()


