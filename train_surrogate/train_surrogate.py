import copy
import os
import time
import argparse
import torch
import random
from torch.backends import cudnn
import torch
from utils import *
from defense import *
# global set
parser = argparse.ArgumentParser(description='Training Victim Surrogates')

parser.add_argument('--eval_dataset', default='cifar10',choices=['cifar10','cifar100','cub200'], type=str)
parser.add_argument('--transfer_dataset',default='cifar100',choices=['cifar100','cifar10','caltech256','imagenet_cifar10','imagenet_cifar100','imagenet_cub200'])
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
    transfer_data, _ = load_data(args.transfer_dataset, train=True)
    transfer_data_deterministic, _ = load_data(args.transfer_dataset, train=True, deterministic=True)
    logging.info("=> creating eval dataset: {}".format(args.eval_dataset))
    eval_data, num_class = load_data(args.eval_dataset, train=False)
    args.num_class = num_class
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    loss = distillation_loss_clf

    # loading teacher model
    teacher = load_model(args)
    teacher_path = './victim_models/victim_{}_{}.pt'.format(args.eval_dataset,args.seed)
    teacher.load_state_dict(torch.load(teacher_path))
    if args.eval_dataset in ['cub200']:
        teacher = torch.nn.DataParallel(teacher)

    # generating perturbations
    perturbations = generate_perturbations(transfer_data_deterministic, teacher, None, method_no_perturbation,
                                           epsilons=None,
                                           batch_size=args.batch_size, num_workers=20)

    transfer_data = make_distillation_dataset(transfer_data, perturbations)

    # loading surrogate model
    from resnet_meta import MetaResNet34,MetaResNet50
    if args.eval_dataset in ['cifar10','cifar100']:
        student = MetaResNet34(num_classes=args.num_class).cuda()
    elif args.eval_dataset in ['cub200']:
        student = MetaResNet50(num_classes=args.num_class).cuda()
    else:
        raise ValueError('{} is an invalid dataset!'.format(args.eval_dataset))

    train_loader = torch.utils.data.DataLoader(transfer_data, shuffle=True, pin_memory=True,
                                               batch_size=args.batch_size, num_workers=20)

    # optimizing surrogate model
    optimizer = torch.optim.SGD(student.params(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    train_with_distillation(args,student, train_loader, eval_loader, optimizer, loss, num_epochs=args.epochs)




def train_with_distillation(args,model, loader, test_loader, optimizer, loss_fn, num_epochs=50, temperature=1.0,print_every=100):
    logging.info('=> Start Training')
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

        if (epoch+1)%10==0:
            logging.info('\n\n Saving model of epoch {} to surrogate_models'.format(epoch+1))
            if not os.path.isdir('./surrogate_models'):
                os.makedirs('./surrogate_models/')
            torch.save(model.state_dict(), './surrogate_models/surrogate_{}_to_{}_{}_epoch{}.pt'.format(args.transfer_dataset,args.eval_dataset, args.seed,epoch+1))



if __name__ == '__main__':
    main()