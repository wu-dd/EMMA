import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch
from torchvision import models
from resnet import *
import logging
import numpy as np


class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, distillation_targets):
        super().__init__()
        self.dataset = dataset
        self.distillation_targets = distillation_targets
        assert len(dataset) == len(distillation_targets), 'Should have same length'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.distillation_targets[idx]


def make_distillation_dataset(dataset, distillation_targets):
    return ZippedDataset(dataset, distillation_targets)

	
def distillation_loss_clf(logits, distill_target, gt_target, temperature):
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # distillation loss
    target_logits = torch.clamp(distill_target, min=1e-12, max=1).log() / temperature
    target = torch.softmax(target_logits, dim=1)
    distill_loss = -1 * (normalized_logits / temperature * target).sum(1).mean(0)

    # # normal loss
    # normal_loss = F.cross_entropy(logits, gt_target, reduction='mean')

    return distill_loss  # * (temperature ** 2) + normal_loss

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
    return lr

def load_model(args):
    if args.eval_dataset in ['cifar10','cifar100']:
        model = resnet34(num_classes=args.num_class).cuda()
    elif args.eval_dataset in ['cub200']:
        model = models.resnet50(num_classes=1000, pretrained=True)
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], args.num_class)
        model = model.cuda()
    else:
        raise ValueError('{} is an invalid dataset!'.format(args.eval_dataset))
    return model


def load_data(dataset_name, train=True, deterministic=False, seed=1):
    from dataset_cub import Cub2011
    from torchvision.datasets.folder import pil_loader
    if train == False:
        dataset_name = dataset_name.split('_')[0]

    if dataset_name in ['cub200']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        num_classes = 200
        dataset = Cub2011(root='~/data/Paper_data/CUB200',train=train, transform=transform, loader=pil_loader)

    elif dataset_name in ['caltech256']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        num_classes = 257
        dataset = dsets.ImageFolder(root='~/data/Paper_data/CALTECH256',transform=transform)

    elif dataset_name in ['imagenet_cub200']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        split = 'train' if train else 'val'
        num_classes = 200
        dataset = dsets.ImageFolder(f'~/data/Paper_data/ImageNet_CUB200/{split}', transform=transform)
        shuffle_indices = np.arange(len(dataset))
        rng = np.random.RandomState(seed)
        rng.shuffle(shuffle_indices)
        dataset = torch.utils.data.Subset(dataset, shuffle_indices[:30000])  # for comparability with Caltech256


    elif dataset_name in ['imagenet_cifar10', 'imagenet_cifar100']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        split = 'train' if train else 'val'
        if dataset_name in ['imagenet_cifar10']:
            num_classes = 10
            dataset = dsets.ImageFolder(f'~/data/Paper_data/ImageNet_CIFAR10/{split}', transform=transform)
        elif dataset_name in ['imagenet_cifar100']:
            num_classes = 100
            dataset = dsets.ImageFolder(f'~/data/Paper_data/ImageNet_CIFAR100/{split}', transform=transform)

    elif dataset_name in ['cifar10', 'cifar100']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        if dataset_name in ['cifar10']:
            num_classes = 10
            dataset = dsets.CIFAR10('~/data/Paper_data/CIFAR10', train=train, transform=transform, download=False)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
        elif dataset_name in ['cifar100']:
            num_classes = 100
            dataset = dsets.CIFAR100('~/data/Paper_data/CIFAR100', train=train, transform=transform, download=False)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)


    return dataset, num_classes