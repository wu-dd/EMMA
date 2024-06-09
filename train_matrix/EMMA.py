import copy
import os
import argparse
import random
from torch.backends import cudnn
import torch
from utils import *
from defense import *
import torch.nn.functional as F

# global set
parser = argparse.ArgumentParser(description='EMMA')

parser.add_argument('--eval_dataset', default='cifar10',choices=['cifar10','cifar100','cub200'], type=str)
parser.add_argument('--transfer_dataset',default='cifar100',choices=['cifar100','cifar10','caltech256','imagenet_cifar10','imagenet_cifar100','imagenet_cub200'],type=str)
# emma parameter
parser.add_argument('--lamda',default='0.01',type=float)
parser.add_argument('--lr_gamma', default=0.3, type=float, help='lr for T')
parser.add_argument('--st_epoch',default=20,type=int)
# fixed
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=20, type=int)
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

class newDataset(torch.utils.data.Dataset):
    def __init__(self, transfer_dataset, transfer_data_deterministic):
        super().__init__()
        self.transfer_dataset = transfer_dataset
        self.transfer_data_deterministic=transfer_data_deterministic

    def __len__(self):
        return len(self.transfer_dataset)

    def __getitem__(self, idx):
        return self.transfer_dataset[idx], self.transfer_data_deterministic[idx]

class EarlyStopping(object):
    def __init__(self, T, diag_sum,patience=7):
        self.patience = patience
        self.counter = 0
        self.best_diag_sum = diag_sum
        self.best_T=T
        self.early_stop = False

    def __call__(self, diag_sum,T):
        if (diag_sum<=self.best_diag_sum):
            self.best_diag_sum=diag_sum
            self.best_T=T
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def main_worker(args):
    # loading data
    transfer_data, _ = load_data(args.transfer_dataset, train=True)
    transfer_data_deterministic, _ = load_data(args.transfer_dataset, train=True, deterministic=True)
    logging.info("=> creating eval dataset: {}".format(args.eval_dataset))
    eval_data, num_class = load_data(args.eval_dataset, train=False)
    args.num_class = num_class
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    loss = distillation_loss_clf

    # loading teacher model
    teacher = load_model(args)
    teacher_path = './victim_models/victim_{}_{}.pt'.format(args.eval_dataset, args.seed)
    teacher.load_state_dict(torch.load(teacher_path))
    if args.eval_dataset in ['cub200']:
        teacher = torch.nn.DataParallel(teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    # loading matrix
    def gen_random_matrix(n):
        matrix=torch.eye(n,n)
        return matrix

    T = gen_random_matrix(num_class).cuda()
    T.requires_grad = True

    # loading student Model
    from resnet_meta import MetaResNet34, MetaResNet50
    if args.eval_dataset in ['cifar10', 'cifar100']:
        student = MetaResNet34(num_classes=args.num_class).cuda()
    elif args.eval_dataset in ['cub200']:
        student = MetaResNet50(num_classes=args.num_class).cuda()
    else:
        raise ValueError('{} is an invalid dataset!'.format(args.eval_dataset))
    student.load_state_dict(torch.load('./surrogate_models/surrogate_{}_to_{}_{}_epoch{}.pt'.format(args.transfer_dataset,args.eval_dataset,args.seed,args.st_epoch)))

    # loading transfer data, transfer_data for surrogate model, transfer_data_deterministic for teacher model
    all_transfer_data = newDataset(transfer_data, transfer_data_deterministic)
    train_loader = torch.utils.data.DataLoader(all_transfer_data, shuffle=True, pin_memory=True,batch_size=args.batch_size, num_workers=20)

    # optimizing matrix
    emma(args, loss, teacher, student, T, train_loader, eval_loader,args.st_epoch)

def emma(args, loss, teacher, main_net, T, train_loader, test_loader,start_epoch):
    optimizer = torch.optim.SGD(main_net.params(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # loading early stopping mechanism
    T_pre = copy.deepcopy(T.data)
    if args.eval_dataset in ['cifar10']:
        threshold=0.01
    elif args.eval_dataset in ['cifar100','cub200']:
        threshold=0.0001
    early_stopping=EarlyStopping(patience=3,T=T_pre,diag_sum=T_pre.data.diagonal().sum())

    # start optimizing
    for epoch in range(start_epoch,args.epochs):
        def meta_net_lr(args,epoch):
            import math
            lr=args.lr
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / args.epochs)) / 2
            return lr

        adjust_learning_rate(args, optimizer, epoch)
        for i, ((data, _), (data_deterministic, _)) in enumerate(train_loader):
            main_net.train()
            teacher.eval()
            data, data_deterministic = data.cuda(), data_deterministic.cuda()
            meta_net = copy.deepcopy(main_net)

            # attack loss
            logit_s = meta_net(data)
            prob_t = F.softmax(teacher(data_deterministic), dim=1)
            prob_t_T = torch.mm(prob_t, T)
            loss_inner = loss(logit_s, prob_t_T, None, temperature=1.0)
            # update the meta_net virtually
            meta_net.zero_grad()
            grads = torch.autograd.grad(loss_inner, (meta_net.params()), create_graph=True)
            meta_net.update_params(meta_net_lr(args,epoch), source_params=grads)

            # defense loss
            target = torch.max(prob_t, dim=1)[1]
            onehot_target = F.one_hot(target, num_classes=args.num_class)
            loss_align = -(torch.log(prob_t_T) * onehot_target).sum(dim=1).mean()
            loss_mis = -torch.mean(torch.sum(torch.log(1.000001 - F.softmax(meta_net(data), dim=1)) * onehot_target, dim=1))
            loss_outer = args.lamda * loss_align + (1 - args.lamda) * loss_mis
            # update the matrix
            grad_eps = torch.autograd.grad(loss_outer, T, only_inputs=True)[0]
            new_T = torch.clamp(T - args.lr_gamma * grad_eps, min=1e-12)
            norm_c = torch.sum(new_T, 1)
            for j in range(args.num_class):
                if norm_c[j] != 0:
                    new_T[j, :] /= norm_c[j]

            T.data = new_T.data
            # Update main_net
            logit_s = main_net(data)
            prob_t = F.softmax(teacher(data_deterministic), dim=1)
            prob_t_T = torch.mm(prob_t, T).detach()
            loss_inner = loss(logit_s, prob_t_T, None, temperature=1.0)
            optimizer.zero_grad()
            loss_inner.backward()
            optimizer.step()

        logging.info("Epoch {}: {}".format(epoch,test(args, epoch, test_loader, main_net)))

        # Evaluation
        main_net.eval()
        teacher.eval()
        with torch.no_grad():
            stu_acc, perturb_tea_acc = 0, 0
            for itr, (test_img, test_label) in enumerate(test_loader):
                test_img, test_label = test_img.cuda(), test_label.cuda()
                logit_stu = main_net(test_img)
                perturb_prob_tea = torch.mm(F.softmax(teacher(test_img), dim=1), T)
                pre_stu = torch.max(logit_stu, 1)[1]
                perturb_pre_tea = torch.max(perturb_prob_tea, 1)[1]
                perturb_tea_acc = perturb_tea_acc + float(torch.sum(perturb_pre_tea == test_label))
                stu_acc = stu_acc + float(torch.sum(pre_stu == test_label))

            test_stu_acc = (stu_acc / float(test_loader.dataset.__len__())) * 100
            test_perturb_tea_acc = (perturb_tea_acc / float(test_loader.dataset.__len__())) * 100
            logging.info('=======[%d/%d][Evaluation]=======\n'
                         '[Teacher] Test Acc: %.4f%%\n'
                         '[Student] Test Acc: %.4f%%' % (
                         epoch, args.epochs, test_perturb_tea_acc, test_stu_acc))
            logging.info('\n%s', str(T.data))

        # early stopping mechanism
        T_diag=T.data.diagonal().sum()
        early_stopping(T_diag,T)
        logging.info((T.data-T_pre.data).abs().mean())
        if (T.data-T_pre.data).abs().mean()<threshold: # early stopping
            break
        elif early_stopping.early_stop==True:
            T=early_stopping.best_T
            print("Early stopping")
            break

        T_pre = copy.deepcopy(T.data)

    if not os.path.isdir('./matrix/'):
        os.makedirs('./matrix/')
    torch.save(T, './matrix/{}_to_{}_gamma_{}_lamda_{}_STepoch{}_{}.pt'.format(args.transfer_dataset, args.eval_dataset, args.lr_gamma, args.lamda, args.st_epoch, args.seed))

if __name__ == '__main__':
    main()
