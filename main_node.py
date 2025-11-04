import time
import yaml
import argparse
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import r2_score
from model_node import Specformer
from utils import count_parameters, init_params, seed_everything, get_split
import matplotlib.pyplot as plt
import numpy as np

def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)
    torch.cuda.set_device(device)

    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    nlayer = config['nlayer']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    tran_dropout = config['tran_dropout']
    feat_dropout = config['feat_dropout']
    prop_dropout = config['prop_dropout']
    norm = config['norm']


    if 'signal' in args.dataset:
        e, u, x, y, m = torch.load('data/{}.pt'.format(args.dataset))
        e, u, x, y, m = e.cuda(), u.cuda(), x.cuda(), y.cuda(), m.cuda()
        mask = torch.where(m == 1)
        x = x[:, args.image].unsqueeze(1)
        y = y[:, args.image]
    else:
        e, u, x, y = torch.load('data/{}.pt'.format(args.dataset))
        e, u, x, y = e.cuda(), u.cuda(), x.cuda(), y.cuda()

        if len(y.size()) > 1:
            if y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1)

        train, valid, test = get_split(args.dataset, y, nclass, args.seed) 
        train, valid, test = map(torch.LongTensor, (train, valid, test))
        train, valid, test = train.cuda(), valid.cuda(), test.cuda()

    nfeat = x.size(1)
    net = Specformer(nclass,nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, norm).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    print(count_parameters(net))

    res = []
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    evaluation_f1 =torchmetrics.F1Score(task='multiclass', num_classes=nclass)
    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()
        # start_time=time.time()
        logits = net(e, u, x)
        # end_time=time.time()
        # print(end_time-start_time)
        # print(logits.shape)
        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            loss = torch.square((logits[mask] - y[mask])).sum()
        else:
            loss = F.cross_entropy(logits[train], y[train]) 

        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(e, u, x)
        
        # _,labels=torch.max(logits,dim=1)
        # print(labels)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            r2 = 1-r2_score(y[mask].data.cpu().numpy(), logits[mask].data.cpu().numpy())
            sse = torch.square(logits[mask] - y[mask]).sum().item()
            print(r2, sse)
            res.append(r2)
        else:
            train_loss=loss.item()
            train_acc=evaluation(logits[train].cpu(),y[train].cpu()).item()
            val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
            test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()

            res.append(test_acc)

            print(idx, train_loss ,train_acc , val_acc, test_acc)
    if 'signal' in args.dataset:
        print(min(res))
    else:
        print(max(res))
    if 'signal' in args.dataset:
        logits = net(e, u, x)
        logits=logits.cpu().detach().numpy()
        logits=np.reshape(logits,(100,100))
        plt.imshow(logits.T)
        plt.colorbar()
        plt.savefig('./images/true/FreLinear.png')
        y=y.cpu().detach().numpy()
        y=np.reshape(y,(100,100))
        plt.imshow(y.T)
        plt.savefig('./images/false/Ground Truth.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--image', type=int, default=37-1)

    args = parser.parse_args()
    
    if 'signal' in args.dataset:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)['signal']
    else:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    main_worker(args, config)

