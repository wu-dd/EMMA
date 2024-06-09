import argparse
import random
from torch.backends import cudnn
import torch
from utils import *
import pickle
# global set
parser = argparse.ArgumentParser(description='Training Attacker')

parser.add_argument('--eval_dataset', default='cifar10',choices=['cifar10','cifar100','cub200'], type=str)
parser.add_argument('--transfer_dataset',default='cifar100',choices=['cifar100','cifar10','caltech256','imagenet_cifar10','imagenet_cifar100','imagenet_cub200'],type=str)
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
    transfer_data, _ = load_data(args.transfer_dataset, train=True)
    eval_data, num_classes = load_data(args.eval_dataset, train=False)
    args.num_class=num_classes
    test_loader = torch.utils.data.DataLoader(eval_data,batch_size=args.batch_size, num_workers=4,
                                              shuffle=False, pin_memory=True)

    # loading perturbations
    load_path='./perturbations/{}_to_{}_eval_perturbations_False_gamma_{}_lamda_{}_STepoch{}_{}.pkl'.format(
            args.transfer_dataset, args.eval_dataset, args.lr_gamma, args.lamda,args.st_epoch, args.seed)
    with open(load_path, 'rb') as f:
        perturbations_dict = pickle.load(f)
    perturbations = torch.FloatTensor(perturbations_dict)
    transfer_data = make_distillation_dataset(transfer_data, perturbations)

    loss = distillation_loss_clf

    logging.info('\nTraining model with transfer_data: {} with eval_data: {}\n'.format(args.transfer_dataset, args.eval_dataset))

    # loading student model
    student = load_model(args)
    loader = torch.utils.data.DataLoader(transfer_data, shuffle=True, pin_memory=True,
                                         batch_size=args.batch_size, num_workers=20)
    optimizer = torch.optim.SGD(student.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)

    # attack
    train_with_distillation(args,student, loader, test_loader, optimizer, loss, num_epochs=args.epochs, temperature=1.0)

    # logging.info('\n\nDone! Saving model to attacker models')
    # save_path='./attacker_models/attacker_{}_to_{}_{}_gamma_{}_lamda_{}_Startepoch{}_{}.pt'.format(
    #         args.transfer_dataset, args.eval_dataset, args.defense, args.lr_gamma, args.lamda,args.st_epoch, args.seed)
    # torch.save(student.state_dict(),save_path)

def train_with_distillation(args,model, loader, test_loader, optimizer, loss_fn, num_epochs=50, temperature=1.0):
    for epoch in range(num_epochs):
        adjust_learning_rate(args, optimizer, epoch)
        model.train()
        for i, (tmp1, distill_targets) in enumerate(loader):
            bx, by = tmp1
            bx = bx.cuda()
            by = by.cuda()
            distill_targets = distill_targets.cuda()

            logits = model(bx)
            loss = loss_fn(logits, distill_targets, by, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate on validation set
        val_acc = test(args, epoch, test_loader, model)
        logging.info("[Epoch {}]:{:.2f}\t".format(epoch, val_acc))


if __name__ == '__main__':
    main()
