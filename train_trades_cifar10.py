from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from models.wideresnet_update import *
from models.resnet import *
from trades import trades_loss, model_para_count, diff_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=85, metavar='N',
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
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
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
args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.backends.cudnn.benchmark = True
#torch.cuda.set_device(args.local_rank)
#torch.distributed.init_process_group(backend='nccl')
args.batch_size = 32
args.test_batch_size = 32

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
f=open("./cifar10-output/test1.txt","a")
args.beta = 1.3

def train(args, model, device, train_loader, optimizer, epoch, para_count):
    model.train()
    hess = []
    grad = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        # loss, temp, temp1 = trades_loss(model=model,
        #                    x_natural=data,
        #                    y=target,
        #                    optimizer=optimizer,
        #                    step_size=args.step_size,
        #                    epsilon=args.epsilon,
        #                    perturb_steps=args.num_steps,
        #                    beta=args.beta,
        #                    hess_threshold=args.hess_threshold,
        #                    evalu= False)
        loss, temp, temp1 = diff_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           hess_threshold=args.hess_threshold,
                           evalu= False)
        hess.append(temp)
        grad.append(temp1)
        loss.backward()
        optimizer.step()
        
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), flush=True, file=f)
    print('================================================================', flush=True, file=f)
    print('Avg Gradient: {}'.format(np.mean(grad)), flush=True, file=f)
    print('Avg Hessian: {}\n std: {} \n median: {} \n min: {} \n max: {}'.format(
       np.mean(hess)/para_count, np.std(hess)/para_count, np.median(hess)/para_count, 
       min(hess)/para_count, max(hess)/para_count), flush=True, file=f)
    return np.mean(hess)
def eval_train(model, device, train_loader):
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
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
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
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_hess_thre(epoch, avg_hess):
    if epoch >= args.start_epoch:
        args.hess_threshold = avg_hess * 1.5
    if epoch >= args.end_epoch:
        args.hess_threshold = avg_hess * 2

def main():
    # init model, ResNet18() can be also used here for training
    model = WideResNet().to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    hess_list = []
    if len(sys.argv) == 3:
        start = int(sys.argv[1])
        length = int(sys.argv[2])
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
            adjust_hess_thre(epoch, np.mean(hess_list))
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
        para_count = model_para_count(model)
        print(para_count, flush=True, file=f)
        for epoch in range(1, args.epochs + 1):
            # adjust learning rate for SGD
            adjust_learning_rate(optimizer, epoch)
            # adversarial training
            avg = train(args, model, device, train_loader, optimizer, epoch, para_count)
            hess_list.append(avg)
            if(len(hess_list) > 4):
                hess_list.pop(0)
            adjust_hess_thre(epoch, np.mean(hess_list))
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


if __name__ == '__main__':
    main()
