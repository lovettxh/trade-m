import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import sys

'''
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()



def hessian_cal(model, loss):
    g1 = torch.autograd.grad(loss, model.parameters(), create_graph=True, only_inputs=True)
    temp =  [torch.ones_like(t, requires_grad=True) for t in model.parameters()]
    s1 = torch.sum(torch.stack([torch.dot(torch.flatten(x),torch.flatten(y)) for x,y in zip(g1,temp)]))
    g2 = torch.autograd.grad(s1, model.parameters(), create_graph=True, only_inputs=True)
    g2 = [torch.abs(x) for x in g2]
    s2 = torch.sum(torch.stack([torch.dot(torch.flatten(x),torch.flatten(y)) for x,y in zip(g2,temp)]))
    g1_= [torch.abs(x) for x in g1]
    grad = torch.sum(torch.stack([torch.dot(torch.flatten(x),torch.flatten(y)) for x,y in zip(g1_,temp)]))
    return s2, grad 
'''
    
def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                hess_threshold=75000,
                distance='l_inf',
                evalu=False):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_ce = nn.CrossEntropyLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            # x_adv.requires_grad_()
            # with torch.enable_grad():
            #     loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
            #                            F.softmax(model(x_natural), dim=1))
            # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            # x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            # x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # --------------
            # PGD
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = criterion_ce(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == 'l_2':
        print("l_2 !!!!!")
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    
    
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                 F.softmax(model(x_natural), dim=1))
    loss_robust = (1.0/batch_size) * criterion_ce(model(x_adv), y)

    #--------------------
    h, g = hessian_cal(model, loss_robust)
    #--------------------
    if(h <= hess_threshold or evalu):
        loss = loss_natural + beta * loss_robust
    else:
        loss = loss_natural
    # return loss
    return loss, h.item(), g.item()



def diff_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=False):

    criterion_kl = nn.KLDivLoss(size_average=False)
    # criterion_ce = nn.CrossEntropyLoss(size_average=False)
    
    if adversarial:
        model.eval()
        batch_size = len(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for _ in range(perturb_steps):
            # --------------
            # PGD
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                      F.softmax(model(x_natural), dim=1))
                # loss_ce = criterion_ce(model(x_adv), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(x_natural)

        loss_natural = F.cross_entropy(logits, y)
        logits_adv = model(x_adv)
        # loss_robust = (1.0/batch_size) * criterion_ce(logits_adv, y)
        # loss_robust = 0
        loss_adv = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(model(x_natural), dim=1))
        
        # loss = loss_natural + loss_robust * loss_adv * beta
        loss = loss_natural + loss_adv * beta
        return loss
    else:
        optimizer.zero_grad()
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        return loss
    #if(correct >= batch_size * correct_rate):
    #    loss_robust = (1.0/batch_size) * criterion_ce(logits_adv, y)
    #    loss = loss_natural + beta * loss_robust
    #    return loss, True
    #else:
    #    loss_robust = (1.0/batch_size) * criterion_ce(logits_adv, y)
    #    loss = loss_natural + beta * loss_robust
    #    return loss, False
    
    #return loss, h.item(), g.item()