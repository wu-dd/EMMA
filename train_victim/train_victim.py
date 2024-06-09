import copy
import os
import time
import argparse
import torch
import random
import logging
from torch.backends import cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch
from utils import *
import torch.nn.functional as F
# global set
parser = argparse.ArgumentParser(description='Training Victim Models')

parser.add_argument('--dataset', default='cifar10',choices=['cifar10','cifar100','cub200'], type=str)
# fixed
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--wd', default=1e-3, type=float)

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
torch.set_printoptions(linewidth=2000)


def main():
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info(args.__dict__)
    main_worker(args)

def cross_entropy_loss(logits, gt_target):
    return F.cross_entropy(logits, gt_target, reduction='mean')

def main_worker(args):
    # loading data
    logging.info("=> creating loader '{}'".format(args.dataset))
    train_loader, test_loader, num_class = load_loader(args)
    args.num_class = num_class
    net = load_model(args)
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)

    loss = cross_entropy_loss

    logging.info('=> Start Training')
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(train_loader, net, optimizer, loss)
        val_acc = test(args, epoch, test_loader, net)
        logging.info("[Epoch {}]:{:.2f}\t".format(epoch, val_acc))

    if not os.path.isdir('./victim_models/'):
        os.makedirs('./victim_models')
    torch.save(net.state_dict(),'./victim_models/victim_{}_{}.pt'.format(args.dataset, args.seed))

def train(train_loader, net, optimizer,criterion):
    # switch to train mode
    net.train()
    for i, (img, target) in enumerate(train_loader):
        img, target = img.cuda(), target.cuda()
        outputs = net(img)
        loss=criterion(outputs,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()