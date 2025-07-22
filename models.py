import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
    
class MultiGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, nhead, lkalpha, dropout):
        super(MultiGATLayer,self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lkr = lkalpha
        self.nhead = nhead
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim * nhead)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_dim , nhead)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_dim , nhead)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.lkr)
        self.norm = LayerNorm(out_dim)
        
    def forward(self, x, mask, graph):
        nX = x@self.W#n,out*head
        nX = torch.reshape(nX,(nX.shape[0],-1,self.out_dim)).transpose(0,1)#nhead,n,out
        nX = self.norm(nX)
        
        at1 = torch.einsum('kno,ok->kn',nX,self.a1)[:,:,torch.newaxis]
        at2 = torch.einsum('kno,ok->kn',nX,self.a2)[:,:,torch.newaxis]
        kgraph = at1+at2.transpose(-1,-2)
        kgraph = self.leakyrelu(kgraph)
        kgraph = kgraph*graph
        mask_graph = torch.where(mask[torch.newaxis,:,:],kgraph,torch.tensor(-2 ** 15))
        mask_graph = F.softmax(mask_graph,dim=-1)
        attention = F.dropout(mask_graph, self.dropout, training=self.training)
        y = attention@nX+nX#k,n,o
        y = y.transpose(0,1).reshape(x.shape[0],-1)
        return F.elu(y)
    
class Labelprompt(nn.Module):
    def __init__(self, hid, nclass, nhead, lkalpha, dropout):
        super(Labelprompt,self).__init__()
        self.W = nn.Parameter(torch.empty(size=(hid, nclass)))
        nn.init.kaiming_uniform_(self.W.data)
        self.A = nn.Parameter(torch.empty(size=(nhead, nclass, nclass)))
        self.O = nn.Parameter(torch.empty(size=(nhead*nclass, nclass)))
        nn.init.kaiming_uniform_(self.O.data)
        nn.init.kaiming_uniform_(self.A.data)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(lkalpha)
        self.out = nn.Linear(nhead*nclass,nclass)
        nn.init.kaiming_uniform_(self.out.weight)
        self.norm1 = LayerNorm(nclass)
        self.norm2 = LayerNorm(nclass)
        
    def forward(self,x,mask,graph):
        LnX = x@self.W
        nX = F.softmax(LnX,dim=-1)
        # mean = self.A.mean(dim=(1,2), keepdim=True)
        # std = self.A.std(dim=(1,2), keepdim=True)
        # eps = 1e-6
        # A = (self.A - mean) / (std + eps)
        A = self.A
        prompt = nX@A@(nX.T)
        kgraph = (graph+F.tanh(4*prompt)*graph/4)*mask
        K=self.norm1(LnX)
        # mask_graph = torch.where(mask[torch.newaxis,:,:],kgraph,torch.tensor(-2 ** 15))
        # mask_graph = F.softmax(mask_graph,dim=-1)
        # attention = F.dropout(mask_graph, self.dropout, training=self.training)
        y = self.norm2(kgraph@K)+K#k,n,o
        y = y.transpose(0,1).reshape(x.shape[0],-1)
        y = self.leakyrelu(y)
        return self.out(y),LnX
    
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_output, teacher_output, mask):

        # 只选择mask为True的样本
        masked_student = student_output[mask]
        masked_teacher = teacher_output[mask]
        
        if masked_student.size(0) == 0:  # 如果没有样本被mask
            return torch.tensor(0.0, device=student_output.device)
        
        # 计算蒸馏损失
        soft_student = F.log_softmax(masked_student / self.temperature, dim=1)
        soft_teacher = F.softmax(masked_teacher / self.temperature, dim=1)
        
        # 计算KL散度损失
        distillation_loss = F.kl_div(
            soft_student, soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss

class Model(nn.Module):
    def __init__(self, d_in, d_model, nclass, nh1, nh2, alpha, dropout):
        super(Model,self).__init__()
        self.GAT = MultiGATLayer(d_in, d_model, nh1, alpha, dropout)
        self.norm = LayerNorm(d_model*nh1)
        self.kd = KnowledgeDistillationLoss()
        self.pt = Labelprompt(d_model*nh1,nclass,nh2,alpha,dropout)
        
    def forward(self,x,mask,graph,mask_train=None):
        y = self.GAT(x,mask,graph)
        y = self.norm(y)
        y,p = self.pt(y,mask,graph)
        if mask_train==None:
            return y,0
        else:
            kloss = self.kd(y,p,mask_train)
            return y,[0.02*kloss,p]
        
        
class Model2(nn.Module):
    def __init__(self, d_in, d_model, nclass, nh1, nh2, alpha, dropout):
        super(Model2,self).__init__()
        self.GAT = MultiGATLayer(d_in, d_model, nh1, alpha, dropout)
        self.norm = LayerNorm(d_model*nh1)
        self.GCNweight = nn.Linear(d_model*nh1,32)
        self.norm2 = LayerNorm(32)
        self.classfy = nn.Linear(32,nclass)
    def forward(self,x,mask,graph):
        y = self.GAT(x,mask,graph)
        y = self.norm(y)
        y1 = self.GCNweight(y)
        y2 = graph@y1
        y = self.norm2(y2)+self.norm2(y1)
        y = F.relu(y)
        return self.classfy(y)
    
    
