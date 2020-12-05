# -*- coding: utf-8 -*-
from __future__ import division

""" 
Test the ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

__editor__ = "Il-Ji Choi, Vuno. Inc." # test file
__editor_email__ = "choiilji@gmail.com"

import argparse
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

import os
from isonet.models.isonext import CifarISONext
from isonet.utils.config import C

def get_args():
    parser = argparse.ArgumentParser(description='Test ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=10)
    # Checkpoints
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    
    parser.add_argument('--gpu_id_list', type=str, default='', help="gpu id")

    parser.add_argument('--cfg', required=True,
                        help='path to config file', type=str)

    
    args = parser.parse_args()
    return args

def test():

    # define default variables
    args = get_args()# divide args part and call it as function

    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    # C.SOLVER.TRAIN_BATCH_SIZE *= num_gpus
    # C.SOLVER.TEST_BATCH_SIZE *= num_gpus
    # C.SOLVER.BASE_LR *= num_gpus
    C.freeze()

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    state = {k: v for k, v in args._get_kwargs()}

    # prepare test data parts
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    # test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 10
    else:
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 100

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # initialize model and load from checkpoint
    net = CifarISONext(args.cardinality, args.base_width, args.widen_factor)
    loaded_state_dict = torch.load(args.load)
    temp = {}
    for key, val in list(loaded_state_dict.items()):
        if 'module' in key:
            # parsing keys for ignoring 'module.' in keys
            temp[key[7:]] = val
        else:
            temp[key] = val
    loaded_state_dict = temp
    net.load_state_dict(loaded_state_dict)

    # paralleize model 
    device_ids = list(range(args.ngpu))
    if args.ngpu > 1:
        if args.gpu_id_list:
            # device_ids = list(map(int, args.gpu_id_list.split(',')))
            # os.environ['CUDA_VISIBLE_DEVICES']作用是只允许gpu gpu_id_list='3,5'可用,
            # 然后使用Model = nn.DataParallel(Model, device_ids=[0,1])，作用是从可用的两个gpu中搜索第0和第1个位置的gpu。
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id_list
            # net = torch.nn.DataParallel(net, device_ids=device_ids)
        # torch.nn.DataParallel : use all gpus by default
        # net = torch.nn.DataParallel(net)
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    if args.ngpu > 0:
        net.cuda()
   
    # use network for evaluation 
    net.eval()

    # calculation part
    loss_avg = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

            # test loss average
            loss_avg += loss.item()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

    # finally print state dictionary
    print(state)

if __name__=='__main__':
    test()
