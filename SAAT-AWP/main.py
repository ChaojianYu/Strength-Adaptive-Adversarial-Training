import argparse
import logging
import sys
import time
import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils import *
from utils_awp import AdvWeightPerturb

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        index = slice(None,None,None)
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def sa_pgd(model, X, y, alpha, restarts, norm, PGD_delta):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = PGD_delta.cuda()
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        ii = 8/255.
        epsilon_work = 0/255.
        epsilon_max = 9/255.                     # epsilon_max
        while ii < epsilon_max:
            output = model(normalize(X + delta))
            loss = F.cross_entropy(output, y, reduction='none')
            index = torch.where(loss < 1.7)[0]    # rho
            if len(index) == 0:
                break
            ii = ii + 2/255.                      # tau
            if ii < epsilon_max:
                epsilon_work = ii
            else:
                epsilon_work = epsilon_max
            for jj in range(3):                   # K
                output = model(normalize(X + delta))
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index, :, :, :]
                g = grad[index, :, :, :]
                x = X[index, :, :, :]
                if norm == "l_inf":
                    d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon_work, max=epsilon_work)
                elif norm == "l_2":
                    g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                    scaled_g = g/(g_norm + 1e-10)
                    d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon_work).view_as(d)
                d = clamp(d, lower_limit - x, upper_limit - x)
                delta.data[index, :, :, :] = d
                delta.grad.zero_()
                output = model(normalize(X + delta))
                loss = F.cross_entropy(output, y, reduction='none')
                index = torch.where(loss < 1.7)[0]    # rho
                if len(index) == 0:
                    break
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise')
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model_AWP_SAAT_max9_min17', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chkpt-iters', default=1000, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    
    dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
        proxy = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    proxy = nn.DataParallel(proxy).cuda()

    params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs

    def lr_schedule(t):
        if t / args.epochs < 0.5:
            return args.lr_max
        elif t / args.epochs < 0.75:
            return args.lr_max / 10.
        else:
            return args.lr_max / 100.
    
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    best_test_robust_acc = 0
    logger.info('Epoch \t Train Time \t Test Time \t LR')
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                # Random initialization
                delta = attack_pgd(model, X, y, 8/255., pgd_alpha, 10, args.restarts, args.norm)
                delta = delta.detach()
                delta = sa_pgd(model, X, y, 2/255., args.restarts, args.norm, PGD_delta=delta)
                delta = delta.detach()
            # Natural training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            model.train()

            if epoch >= args.awp_warmup:
                awp = awp_adversary.calc_awp(inputs_adv=X_adv, targets=y)
                awp_adversary.perturb(awp)

            robust_output = model(X_adv)
            
            robust_loss = nn.CrossEntropyLoss(reduce=False)(robust_output, y)
            robust_loss = robust_loss.mean()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)

        train_time = time.time()

        model.eval()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, batch in enumerate(train_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, 8. / 255., pgd_alpha, 10, args.restarts, args.norm)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)


        test_loss_0 = 0
        test_acc_0 = 0
        #test_loss_2 = 0
        #test_acc_2 = 0
        #test_loss_4 = 0
        #test_acc_4 = 0
        #test_loss_6 = 0
        #test_acc_6 = 0
        test_loss_8 = 0
        test_acc_8 = 0
        #test_loss_10 = 0
        #test_acc_10 = 0
        #test_loss_12 = 0
        #test_acc_12 = 0
        #test_loss_14 = 0
        #test_acc_14 = 0
        #test_loss_16 = 0
        #test_acc_16 = 0
        #test_loss_18 = 0
        #test_acc_18 = 0
        #test_loss_20 = 0
        #test_acc_20 = 0
        #test_loss = 0
        #test_acc = 0
        test_n = 0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                #delta_2 = attack_pgd(model, X, y, 2. / 255., pgd_alpha, 5, args.restarts, args.norm)
                #delta_4 = attack_pgd(model, X, y, 4. / 255., pgd_alpha, 10, args.restarts, args.norm)
                #delta_6 = attack_pgd(model, X, y, 6. / 255., pgd_alpha, 15, args.restarts, args.norm)
                delta_8 = attack_pgd(model, X, y, 8. / 255., pgd_alpha, 20, args.restarts, args.norm)
                #delta_10 = attack_pgd(model, X, y, 10. / 255., pgd_alpha, 25, args.restarts, args.norm)
                #delta_12 = attack_pgd(model, X, y, 12. / 255., pgd_alpha, 30, args.restarts, args.norm)
                #delta_14 = attack_pgd(model, X, y, 14. / 255., pgd_alpha, 35, args.restarts, args.norm)
                #delta_16 = attack_pgd(model, X, y, 16. / 255., pgd_alpha, 40, args.restarts, args.norm)
                #delta_18 = attack_pgd(model, X, y, 18. / 255., pgd_alpha, 45, args.restarts, args.norm)
                #delta_20 = attack_pgd(model, X, y, 20. / 255., pgd_alpha, 50, args.restarts, args.norm)
                #delta = attack_pgd(model, X, y, 8. / 255., pgd_alpha, 10, args.restarts, args.norm)
            #delta_2 = delta_2.detach()
            #delta_4 = delta_4.detach()
            #delta_6 = delta_6.detach()
            delta_8 = delta_8.detach()
            #delta_10 = delta_10.detach()
            #delta_12 = delta_12.detach()
            #delta_14 = delta_14.detach()
            #delta_16 = delta_16.detach()
            #delta_18 = delta_18.detach()
            #delta_20 = delta_20.detach()
            #delta = delta.detach()

            #robust_output_2 = model(normalize(torch.clamp(X + delta_2[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_2 = criterion(robust_output_2, y)
            #robust_output_4 = model(normalize(torch.clamp(X + delta_4[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_4 = criterion(robust_output_4, y)
            #robust_output_6 = model(normalize(torch.clamp(X + delta_6[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_6 = criterion(robust_output_6, y)
            robust_output_8 = model(normalize(torch.clamp(X + delta_8[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss_8 = criterion(robust_output_8, y)
            #robust_output_10 = model(normalize(torch.clamp(X + delta_10[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_10 = criterion(robust_output_10, y)
            #robust_output_12 = model(normalize(torch.clamp(X + delta_12[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_12 = criterion(robust_output_12, y)
            #robust_output_14 = model(normalize(torch.clamp(X + delta_14[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_14 = criterion(robust_output_14, y)
            #robust_output_16 = model(normalize(torch.clamp(X + delta_16[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_16 = criterion(robust_output_16, y)
            #robust_output_18 = model(normalize(torch.clamp(X + delta_18[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_18 = criterion(robust_output_18, y)
            #robust_output_20 = model(normalize(torch.clamp(X + delta_20[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss_20 = criterion(robust_output_20, y)
            #robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            #robust_loss = criterion(robust_output, y)

            output_0 = model(normalize(X))
            loss_0 = criterion(output_0, y)

            #test_loss_2 += robust_loss_2.item() * y.size(0)
            #test_acc_2 += (robust_output_2.max(1)[1] == y).sum().item()
            #test_loss_4 += robust_loss_4.item() * y.size(0)
            #test_acc_4 += (robust_output_4.max(1)[1] == y).sum().item()
            #test_loss_6 += robust_loss_6.item() * y.size(0)
            #test_acc_6 += (robust_output_6.max(1)[1] == y).sum().item()
            test_loss_8 += robust_loss_8.item() * y.size(0)
            test_acc_8 += (robust_output_8.max(1)[1] == y).sum().item()
            #test_loss_10 += robust_loss_10.item() * y.size(0)
            #test_acc_10 += (robust_output_10.max(1)[1] == y).sum().item()
            #test_loss_12 += robust_loss_12.item() * y.size(0)
            #test_acc_12 += (robust_output_12.max(1)[1] == y).sum().item()
            #test_loss_14 += robust_loss_14.item() * y.size(0)
            #test_acc_14 += (robust_output_14.max(1)[1] == y).sum().item()
            #test_loss_16 += robust_loss_16.item() * y.size(0)
            #test_acc_16 += (robust_output_16.max(1)[1] == y).sum().item()
            #test_loss_18 += robust_loss_18.item() * y.size(0)
            #test_acc_18 += (robust_output_18.max(1)[1] == y).sum().item()
            #test_loss_20 += robust_loss_20.item() * y.size(0)
            #test_acc_20 += (robust_output_20.max(1)[1] == y).sum().item()
            test_loss_0 += loss_0.item() * y.size(0)
            test_acc_0 += (output_0.max(1)[1] == y).sum().item()
            #test_loss += robust_loss.item() * y.size(0)
            #test_acc += (robust_output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t\t %.4f \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                #test_loss/test_n, test_acc/test_n,
                test_loss_0/test_n, test_acc_0/test_n,
                #test_loss_2/test_n, test_acc_2/test_n,
                #test_loss_4/test_n, test_acc_4/test_n, test_loss_6/test_n, test_acc_6/test_n,
                test_loss_8/test_n, test_acc_8/test_n
                #test_loss_10/test_n, test_acc_10/test_n,
                #test_loss_12/test_n, test_acc_12/test_n, test_loss_14/test_n, test_acc_14/test_n,
                #test_loss_16/test_n, test_acc_16/test_n, test_loss_18/test_n, test_acc_18/test_n,
                #test_loss_20/test_n, test_acc_20/test_n
                )

            # save checkpoint
            if test_acc_8/test_n > best_test_robust_acc:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_acc_8/test_n
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))

        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t %.4f \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss_0/test_n, test_acc_0/test_n, test_loss_2/test_n, test_acc_2/test_n,
                test_loss_4/test_n, test_acc_4/test_n, test_loss_6/test_n, test_acc_6/test_n,
                test_loss_8/test_n, test_acc_8/test_n, test_loss_10/test_n, test_acc_10/test_n,
                test_loss_12/test_n, test_acc_12/test_n, test_loss_14/test_n, test_acc_14/test_n,
                test_loss_16/test_n, test_acc_16/test_n, test_loss_18/test_n, test_acc_18/test_n,
                test_loss_20/test_n, test_acc_20/test_n)
            return


if __name__ == "__main__":
    main()
