# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.model import CifarResNeXt
import datetime
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--gpu_id_list', type=str, default='', help="gpu id")
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    args = parser.parse_args()

    nextline = '\n'
    starttime = datetime.datetime.now()
    ttuple = starttime.timetuple()
    startt = '_'.join(list(map(str, ttuple[1:6])))

    # save model name
    modeloutput = f'{args.dataset}_resnext_{startt}_model.pytorch'

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(
        args.log, f'resnext_train_{args.dataset}_{args.ngpu}gpu_{startt}.txt'), 'w')

    # log.write(starttime)
    log.write(f'{starttime}{nextline}')

    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * 128 // args.batch_size
    args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 10
    else:
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
    log.write(f'{net}{nextline}')
    log.flush()

    device_ids = list(range(args.ngpu))
    if args.ngpu > 1:
        if args.gpu_id_list:
            # device_ids = list(map(int, args.gpu_id_list.split(',')))
            # os.environ['CUDA_VISIBLE_DEVICES']作用是只允许gpu gpu_id_list='3,5'可用,
            # 然后使用Model = nn.DataParallel(Model, device_ids=[0,1])，作用是从可用的两个gpu中搜索第0和第1个位置的gpu。
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id_list
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    if args.ngpu > 0:
        # choose gpu to load model,defalt cuda:0
        net.cuda()
        # to solve warning :module must have its parameters and buffers on device cuda:3 (device_ids[0]) but found one of them on device: cuda:0
        # first:not validate
        # net.cuda(device=device_ids[0])
        # second:
        # device = torch.device(f'cuda:{device_ids[0]}')
        # net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8

        state['train_loss'] = loss_avg


    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # test loss average
            loss_avg += float(loss)

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)


    # Main loop

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        epochstarttime = datetime.datetime.now()

        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net.state_dict(), os.path.join(
                args.save, modeloutput))
        log.write('%s\n' % json.dumps(state))
        # print(state)
        log.write("Best accuracy: %f\n" % best_accuracy)

        epochendtime = datetime.datetime.now()
        log.write(
            f'end: {epochendtime}; len: {epochendtime - epochstarttime}{nextline}')
        log.flush()

    endtime = datetime.datetime.now()
    log.write(
        f'end: {endtime}; len: {endtime - starttime}{nextline}')
    log.flush()
    log.close()
