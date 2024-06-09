import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch
from augment.randaugment import RandomAugment
from torchvision import models
from resnet import *
import logging


def test(args, epoch, test_loader, model):
    with torch.no_grad():
        model.eval()
        top1_acc = AverageMeter("Top1")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])

    return top1_acc.avg

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def adjust_learning_rate(args, optimizer, epoch):
    import math
    lr = args.lr

    ##task cosine
    eta_min=lr * (args.lr_decay_rate**3)
    lr=eta_min+(lr-eta_min)*(
        1+math.cos(math.pi*epoch/args.epochs))/2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    logging.info('LR: {}'.format(lr))

def load_model(args):
    if args.dataset in ['cifar10','cifar100']:
        model = resnet34(num_classes=args.num_class).cuda()
        logging.info("=> Creating model resnet34!")
    elif args.dataset in ['cub200']:
        model = models.resnet50(num_classes=1000, pretrained=True)
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], args.num_class)
        model = model.cuda()
        logging.info("=> Creating model resnet50!")
    else:
        raise ValueError('{} is an invalid dataset!'.format(args.dataset))
    return model



def load_loader(args):
    if args.dataset == "cifar10":
        num_classes = 10
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = dsets.CIFAR10(root='~/data/Paper_data/CIFAR10', train=True, download=True,transform=train_transform)
        test_dataset = dsets.CIFAR10(root='~/data/Paper_data/CIFAR10', train=False, transform=test_transform)

    elif args.dataset == 'cifar100':
        num_classes = 100
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = dsets.CIFAR100(root='~/data/Paper_data/CIFAR100', train=True, download=True, transform=train_transform)
        test_dataset = dsets.CIFAR100(root='~/data/Paper_data/CIFAR100', train=False, transform=test_transform)
    elif args.dataset=='cub200':
        from dataset_cub import Cub2011
        from torchvision.datasets.folder import pil_loader
        num_classes=200
        test_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = Cub2011('~/data/Paper_data/CUB200', train=True, transform=train_transform, loader=pil_loader)
        test_dataset = Cub2011('~/data/Paper_data/CUB200', train=False, transform=test_transform, loader=pil_loader)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
        drop_last=True
    )
    return train_loader,test_loader,num_classes