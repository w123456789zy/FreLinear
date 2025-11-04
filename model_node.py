import time
import math
from math import sqrt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from mamba_ssm import Mamba


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)

class FFTEncoding(nn.Module):
    def __init__(self,hidden_dim=128):
        super().__init__()
        self.lr=nn.Linear(2,hidden_dim)
    def forward(self,x):
        x=torch.fft.fft(x)
        amplitude=torch.abs(x).unsqueeze(0)
        phase=torch.angle(x).unsqueeze(0)
        x=torch.cat((amplitude,phase),dim=0)
        x=self.lr(x.permute(1,0))
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class SpecLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none': 
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':    # Arxiv
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':  # Penn
            self.norm = nn.BatchNorm1d(ncombines)
        else:                  # Others
            self.norm = None 

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x

class LinearAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(LinearAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.elu = nn.ELU()
        self.glu = nn.GELU()
        self.conv=nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.sigmod=nn.Sigmoid()
        self.linear1=nn.Linear(dim_q,dim_q)
        self.linear2=nn.Linear(dim_q,dim_q)
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
        self.out=nn.Linear(dim_q,dim_q)

    def forward(self, x):

        b, n, dim_q = x.shape
        assert dim_q == self.dim_q
        y = self.sigmod(self.linear2(x))
        x = self.glu(self.conv(self.linear1(x)))
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        dist = torch.bmm(v, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.bmm(dist, q)
        att = self.out(att*y)+x
        return att

class Block1(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        # self.atten=LinearAttention(dim,dim,dim)
        self.atten=Mamba(d_model=dim)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=FeedForwardNetwork(dim,dim*2,dim)
    def forward(self,x):
        x1=self.norm1(x)
        x1=self.atten(x)
        x1=x1+x
        x2=self.norm2(x1)
        x2=self.mlp(x2)
        return x2+x1


class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(Specformer, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )
        
        # for arxiv & penn
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.eig_encoder_fft=FFTEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        # self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.mha=Block1(dim=hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none':
            self.layers = nn.ModuleList([SpecLayer(nheads+1, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            self.layers = nn.ModuleList([SpecLayer(nheads+1, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])
        

    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)   # [N,num_feature] => [N,num_class]
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        # eig = self.eig_encoder(e)   # [N, d]
        eig = self.eig_encoder(e)+self.eig_encoder_fft(e)   # [N, d]

        mha_eig = self.mha_norm(eig)
        # mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        mha_eig = mha_eig.unsqueeze(0)
        mha_eig = self.mha(mha_eig)[0]

        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)   # [N, m]

        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h # [N,N]@ [N,num_class] =>[N,num_class]  
            for i in range(self.nheads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]    # 一次图卷积    a=stack(u@(e[:,i]*(u@h)))
            basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]                
            h = conv(basic_feats)                                                                    # Norm(Σ(a*w))

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h