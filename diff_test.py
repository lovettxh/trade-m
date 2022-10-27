from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet_update import *
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--num',default=0,type=int)
parser.add_argument('--auto-attack',default=False)
parser.add_argument('--log-path',default='./log_file.txt')
args = parser.parse_args()


# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def _pgd_whitebox(model_nat,
                  model_adv,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out_nat = model_nat(X)
    out_adv = model_adv(X)
    err_nat = (out_nat.data.max(1)[1] != y.data).float().sum()
    err_adv = (out_adv.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_nat(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model_adv(X_pgd).data.max(1)[1] != y.data).float().sum()
    # label = logit_calculate(model_nat(X_pgd), model_adv(X_pgd))
    # err_pgd = (label.data != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err_nat, err_adv, err_pgd

def eval_adv_test_whitebox(model_nat, model_adv, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model_nat.eval()
    model_adv.eval()
    robust_err_total = 0
    nat_nat_err_total = 0
    nat_adv_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_adv, err_robust = _pgd_whitebox(model_nat, model_adv, X, y)
        robust_err_total += err_robust
        nat_nat_err_total += err_natural
        nat_adv_err_total += err_adv
    print('natural_nat_err_total: ', nat_nat_err_total)
    print('natural_adv_err_total: ', nat_adv_err_total)
    print('robust_err_total: ', robust_err_total)

def logit_calculate(logit_old, logit_new):
    logit_diff = logit_new - logit_old
    label_diff = logit_diff.data.max(1)[1]
    return label_diff

def main():
    model_path_adv = './model-cifar-wideResNet/wideres-adv-epoch76.pt'
    model_path_nat = './model-cifar-wideResNet/wideres-nat-epoch76.pt'
    model_adv = WideResNet().to(device)
    model_nat = WideResNet().to(device)
    model_adv.load_state_dict(torch.load(model_path_adv))
    model_nat.load_state_dict(torch.load(model_path_nat))
    eval_adv_test_whitebox(model_nat, model_adv, device, test_loader)

if __name__ == '__main__':
    main()