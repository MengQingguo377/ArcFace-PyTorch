from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *


def save_model(model, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + f'{iter_cnt:08d}' + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def save_checkpoint(checkpoint_dict, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + f'{iter_cnt:08d}' + '.pth')
    torch.save(checkpoint_dict, save_name)
    return save_name



if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda:3")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    # print(model)
    model.to(device)
    # model = DataParallel(model)
    metric_fc.to(device)
    # metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    epoch_cnt = 0
    #######ADD:load checkpoint#########
    if opt.load_saved_model and os.path.exists(opt.checkpoints_path):
        if os.listdir(opt.checkpoint_path).__len__() != 0:
            checkpoint_name = sorted(os.listdir(opt.checkpoint_path))[-1]
            checkpoint = torch.load(os.path.joint(opt.checkpoints_path, checkpoint_name))
            model.load_state_dict(checkpoint['model'])
            metric_fc.load_state_dict(checkpoint['metric_fc'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch_cnt = checkpoint['epoch_cnt']

    start = time.time()
    for i in range(epoch_cnt + 1, opt.max_epoch + 1):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input) # [B, 512]
            output = metric_fc(feature, label) # [B, num_classes]
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = (i - 1) * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                log = '{} train epoch {}/{} iter {} {} iters/s loss {} acc {}'.format(time_str, i, opt.max_epoch, ii, speed, loss.item(), acc)
                with open(opt.log_path, 'a') as f:
                    f.write(log + '\n')
                print(log)
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if (i - 1) % opt.save_interval == 0 or i == opt.max_epoch:
            checkpoint_dict = {'model': model.state_dict(), 
                               'metric_fc': metric_fc.state_dict(), 
                               'optimizer': optimizer.state_dict(), 
                               'scheduler': scheduler.state_dict(),
                               'epoch_cnt': i}
            save_checkpoint(checkpoint_dict, opt.checkpoints_path, opt.backbone, i)
            # save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')