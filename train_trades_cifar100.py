from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import wandb
from models.wideresnet_update import *
from models.resnet import *
from trades import trades_loss, model_para_count, diff_loss

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
wandb.init(project="trade-loss_monitor", entity="lovettxh", name="cifar100-trade-onestop")

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--hess-threshold', default=0,
                    help='hessian threshold')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--start-epoch', default=10)
parser.add_argument('--end-epoch', default=60)
parser.add_argument('--continue-train',default=0, type=int)
parser.add_argument('--continue-train-len',default=0, type=int)
parser.add_argument('--correct',default=0.8, type=float)
args = parser.parse_args()

wandb.config.update({
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "beta": args.beta
})
wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")
# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.backends.cudnn.benchmark = True


# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
f=open("./cifar100-output/temp.txt","a")

def train(args, model, device, train_loader, optimizer, epoch, para_count):
    model.train()
    true_prob = []
    accur = []
    count = 0
    loss_nat = 0
    loss_rob = 0
    loss_adv = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss, t, tt, nat, rob, adv = diff_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           hess_threshold=args.hess_threshold,
                           correct_rate=args.correct,
                           flag= False)
        loss_nat += nat.item()
        # loss_rob += rob.item()
        loss_adv += adv.item()
        true_prob.append(torch.mean(t).item())
        accur.append(tt)

        loss.backward()
        optimizer.step()
        
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), flush=True, file=f)
    wandb.log({"train trade loss nat": loss_adv})   
    print('================================================================', flush=True, file=f)
    print('loss nat = {}, loss rob = {}, loss adv = {}'.format(loss_nat, loss_rob, loss_adv), flush=True, file=f)  
    print('true_prob = {}, accur = {}'.format(np.mean(true_prob), np.mean(accur)), flush=True, file=f)  


def eval_train(model, device, train_loader,epoch):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)), flush=True, file=f)
    
    training_accuracy = correct / len(train_loader.dataset) * 100.
    wandb.log({"train accuracy nat": training_accuracy})
    # return train_loss, training_accuracy


def eval_test(model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True, file=f)
    test_accuracy = correct / len(test_loader.dataset) * 100.
    wandb.log({"test accuracy nat": test_accuracy})
    # return test_loss, test_accuracy

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    X_pgd = Variable(X.data, requires_grad=True)
    
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_adv = model(X_pgd)
    err_pgd = (out_adv.data.max(1)[1] == y.data).float().sum()
    trade_loss = nn.KLDivLoss(size_average=False)(F.log_softmax(out_adv, dim=1),
                                                     F.softmax(out, dim=1))
    return err_pgd, trade_loss

def eval_adv_test_whitebox(model, device, test_loader,epoch, train=True):
    """
    evaluate model by white-box attack
    """
    model.eval()
    t_loss = 0
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust, loss = _pgd_whitebox(model, X, y)
        t_loss += loss.item()
        robust_err_total += err_robust
    t_loss /= len(test_loader.dataset)
    if train:
        wandb.log({"train accuracy adv": robust_err_total/len(test_loader.dataset)*100.})
        wandb.log({"train trade loss adv": t_loss})
    else:
        wandb.log({"test accuracy adv": robust_err_total/len(test_loader.dataset)*100.})
        wandb.log({"test trade loss adv": t_loss})

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 70:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # init model, ResNet18() can be also used here for training
    model = WideResNet().to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    hess_list = []
    para_count = model_para_count(model)
    if args.continue_train != 0:
        start = args.continue_train
        length = args.continue_train_len
        model_path = "./model-cifar-wideResNet/model-wideres-epoch"+ str(start) +".pt"
        optimizer_path = "./model-cifar-wideResNet/opt-wideres-checkpoint_epoch" + str(start) + ".tar"
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("load complete")
        for epoch in range(start+1, length+start+1):
            adjust_learning_rate(optimizer, epoch)
            avg = train(args, model, device, train_loader, optimizer, epoch, para_count)
            hess_list.append(avg)
            if(len(hess_list) > 4):
                hess_list.pop(0)
            # adjust_hess_thre(epoch, np.mean(hess_list))
            # evaluation on natural examples
            print('================================================================', flush=True, file=f)
            eval_train(model, device, train_loader)
            eval_test(model, device, test_loader)
            print('================================================================', flush=True, file=f)
            # save checkpoint
            if epoch % args.save_freq == 0:
                torch.save(model.state_dict(),
                        os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
                torch.save(optimizer.state_dict(),
                        os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
    else:
        
        print(para_count, flush=True, file=f)
        for epoch in range(1, args.epochs + 1):
            wandb.log({"epoch":epoch})
            # adjust learning rate for SGD
            adjust_learning_rate(optimizer, epoch)
            # adversarial training
            train(args, model, device, train_loader, optimizer, epoch, para_count)
            #hess_list.append(avg)
            #if(len(hess_list) > 4):
            #    hess_list.pop(0)
            # adjust_hess_thre(epoch, np.mean(hess_list))
            # evaluation on natural examples
            print('================================================================', flush=True, file=f)
            eval_train(model, device, train_loader, epoch)
            eval_test(model, device, test_loader, epoch)
            if epoch % 3 == 0:
                eval_adv_test_whitebox(model, device, train_loader,epoch, True)
                eval_adv_test_whitebox(model, device, test_loader,epoch, False)
            print('================================================================', flush=True, file=f)

            # save checkpoint
            if epoch % args.save_freq == 0:
                torch.save(model.state_dict(),
                        os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
                torch.save(optimizer.state_dict(),
                        os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
