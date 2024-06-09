import argparse
import random
from torch.backends import cudnn
import torch
from utils import *
from defense import *
import pickle


parser = argparse.ArgumentParser(description='Eval Victim Models')

parser.add_argument('--eval_dataset', default='mnist',choices=['mnist','cifar10','cifar100','cub200'], type=str)
parser.add_argument('--transfer_dataset',default='svhn',choices=['fashionmnist','cifar100','cifar10','caltech256','qmnist','imagenet_cifar10','imagenet_cifar100','imagenet_cub200'],type=str)
parser.add_argument('--eval_perturbations', action='store_true', help='if true, generate perturbation val queries directly for eval teacher')
##
parser.add_argument('--lamda',default='3.0',type=float)
parser.add_argument('--lr_gamma', default=1e-2, type=float, help='lr for T')
parser.add_argument('--st_epoch',default=10,type=int)
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
torch.set_printoptions(linewidth=2000,sci_mode=False,precision=3)

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

def main_worker(args):
    # loading data
    logging.info("=> creating transfer dataset: {}".format(args.transfer_dataset))
    transfer_data, _ = load_data(args.eval_dataset, train=False)
    logging.info("=> creating eval dataset: {}".format(args.eval_dataset))
    eval_data, num_class = load_data(args.eval_dataset, train=False)
    args.num_class = num_class

    # loading perturbations
    load_path = './perturbations/{}_to_{}_eval_perturbations_True_gamma_{}_lamda_{}_STepoch{}_{}.pkl'.format(
            args.transfer_dataset, args.eval_dataset, args.lr_gamma, args.lamda,args.st_epoch, args.seed)
    with open(load_path, 'rb') as f:
        perturbations_dict = pickle.load(f)
    perturbations = torch.FloatTensor(perturbations_dict)
    transfer_data = make_distillation_dataset(transfer_data, perturbations)
    loader = torch.utils.data.DataLoader(transfer_data, shuffle=True, pin_memory=True,
                                         batch_size=args.batch_size, num_workers=20)

    # evaluation
    top1_acc = AverageMeter("Top1")
    for i, (tmp1, distill_targets) in enumerate(loader):
        bx, by = tmp1
        bx = bx.cuda()
        by = by.cuda()
        distill_targets = distill_targets.cuda()
        acc1, _ = accuracy(distill_targets, by, topk=(1, 5))
        top1_acc.update(acc1[0])

    logging.info("Teacher accuracy  :{:.2f}\t".format(top1_acc.avg))

if __name__ == '__main__':
    main()

