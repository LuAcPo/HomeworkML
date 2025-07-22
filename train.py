import torch
import numpy as np
import models
import torch.nn.functional as F
import time
import torch.nn as nn
from torch import optim
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def GetData(temperature = 2):
    data = np.load('./cora_numpy/features.npy')
    mask = np.load('./data/mask.npy')
    graph = np.load('./data/dataT'+str(temperature)+'.npy')
    labels = np.load('./cora_numpy/labels.npy')
    return torch.from_numpy(data).float(), torch.from_numpy(mask
            ).bool(), torch.from_numpy(graph).float(), torch.from_numpy(labels)
def getrawData():
    mask = np.load('./data/d0.npy')
    graph = np.load('./data/d0.npy')
    return torch.from_numpy(mask).bool(
        ),torch.from_numpy(graph).float()
    
def evalmodel(data,mask,graph,labels,index_test,path):
    model = torch.load(path)
    model.eval()
    output,_ = model(data,mask,graph)
    acc = accuracy(output[index_test],labels[index_test])
    print("final acc:",acc.item())
    
def two(data,mask,graph,labels,index_test,temperature):
    path = 'T'+str(temperature)+'best_val_model.pkl'
    evalmodel(data,mask,graph,labels,index_test,path)
    # path = 'T'+str(temperature)+'model_120.pkl'
    # evalmodel(data,mask,graph,labels,index_test,path)
    
def main(index_train,index_val,index_test,temperature=2, if_eval=False):
    
    data,mask,graph,labels = GetData(temperature)
    # data,_,_,labels = GetData(temperature)
    # mask,graph = getrawData()
    model = models.Model(d_in=data.shape[-1], d_model=8, nclass=7, nh1=8, nh2=8, alpha=0.2, dropout=0.6)
    
    if(if_eval):
        two(data,mask,graph,labels,index_test,temperature)
        return
    opt = optim.Adam(model.parameters(),lr=0.005,weight_decay=0.3)
    
    best_val_acc = 0
    # best_epoch = 50
    min_val_loss = 120.
    for epoch in range(150):
        opt.zero_grad()
        output,aloss = model(data,mask,graph,index_train)
        loss_kd = aloss[0]
        
        train_acc = accuracy(output[index_train],labels[index_train])
        val_acc = accuracy(output[index_val],labels[index_val])
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            torch.save(model,'T'+str(temperature)+'best_val_model.pkl')
        elif val_acc == best_val_acc and lossval<min_val_loss:
            min_val_loss = lossval
            torch.save(model,'T'+str(temperature)+'best_val_model.pkl')
        lossval = F.cross_entropy(output[index_val],labels[index_val])
        
        loss = F.cross_entropy(output[index_train],labels[index_train]
                               )+loss_kd+0.05*F.cross_entropy(aloss[1][index_train],labels[index_train])
        loss.backward()

        opt.step()
        print(f'epoch: {epoch},train_acc: {round(train_acc.item(),5)} ,train_loss: {round(loss.item(),5)}')
        
        print(f'val_acc: {round(val_acc.item(),5)} ,loss_val: {round(lossval.item(),5)}')

            
        print('=========================')
    two(data,mask,graph,labels,index_test,temperature)
    

if __name__ == '__main__':
    idx_train = np.where(np.load('cora_numpy/mk_train.npy')==True)
    idx_val = np.where(np.load('cora_numpy/mk_val.npy')==True)
    idx_test = np.where(np.load('cora_numpy/mk_test.npy')==True)
    idx_train = torch.LongTensor(idx_train)[0]
    idx_val = torch.LongTensor(idx_val)[0]
    idx_test = torch.LongTensor(idx_test)[0]
    if_eval = False
    main(idx_train,idx_val,idx_test,1,if_eval)