import copy
import os
import time
import argparse
import torch
import random
import logging
from torch.backends import cudnn
import torch
from utils import *
from defense import *
import pickle
# global set
parser = argparse.ArgumentParser(description='Generating query')

parser.add_argument('--eval_dataset', default='cifar10',choices=['cifar10','cifar100','cub200'], type=str)
parser.add_argument('--transfer_dataset',default='cifar100',choices=['cifar100','cifar10','caltech256','imagenet_cifar10','imagenet_cifar100','imagenet_cub200'],type=str)
parser.add_argument('--eval_perturbations', action='store_true', help='if true, generate perturbations on val set of transfer dataset')
##
parser.add_argument('--lamda',default=3.0,type=float)
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
    os.environ["NUMEXPR_MAX_THREADS"] = "32"
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
    transfer_data, _ = load_data(args.eval_dataset if args.eval_perturbations else args.transfer_dataset,
                                 train=not args.eval_perturbations, deterministic=True)
    logging.info("=> creating eval dataset: {}".format(args.eval_dataset))
    eval_data, num_class = load_data(args.eval_dataset, train=False)
    args.num_class = num_class

    # loading teacher model
    teacher = load_model(args)
    teacher_path = './victim_models/victim_{}_{}.pt'.format(args.eval_dataset, args.seed)
    teacher.load_state_dict(torch.load(teacher_path))
    teacher.eval()

    # loading matrix
    T=torch.load('./matrix/{}_to_{}_gamma_{}_lamda_{}_STepoch{}_{}.pt'.format(args.transfer_dataset, args.eval_dataset, args.lr_gamma, args.lamda,args.st_epoch,  args.seed))
    T=T.cuda()

    # generating perturbations
    shuffle_indices = np.arange(len(transfer_data))
    np.random.shuffle(shuffle_indices)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(transfer_data, shuffle_indices),
                                         batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)
    teacher_pred_perturbed = []
    for (bx, by) in tqdm.tqdm(loader, mininterval=1.0):
        bx = bx.cuda()
        with torch.no_grad():
            teacher_logits = teacher(bx)
            teacher_pred = torch.mm(torch.softmax(teacher_logits, dim=1),T)
            out=teacher_pred.detach()
        teacher_pred_perturbed.append(out)
    teacher_pred_perturbed = torch.cat(teacher_pred_perturbed, dim=0)
    unshuffle_indices = np.zeros(len(transfer_data))
    for i, p in enumerate(shuffle_indices):
        unshuffle_indices[p] = i
    perturbations = teacher_pred_perturbed[unshuffle_indices]

    # saving perturbations data
    perturbations_dict = perturbations.data.cpu().numpy()
    if not os.path.isdir('perturbations'):
        os.makedirs('perturbations')
    save_path = './perturbations/{}_to_{}_eval_perturbations_{}_gamma_{}_lamda_{}_STepoch{}_{}.pkl'.format(
            args.transfer_dataset, args.eval_dataset, args.eval_perturbations, args.lr_gamma, args.lamda,args.st_epoch, args.seed)
    with open(save_path, 'wb') as f:
        pickle.dump(perturbations_dict, f)



if __name__ == '__main__':
    main()