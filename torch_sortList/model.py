#!/usr/bin/env python
# coding: utf-8

# In[70]:


import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


# In[71]:


class cls_data(Dataset):
    def __init__(self,nsample=5000,is_train=True):
        
        if is_train == True:
            high = 100
            self.data = np.random.randint(1, high=high, size=(nsample,10))
            self.data[int(nsample/2):].sort()
            np.random.shuffle(self.data)
            
            self.label = np.ones((nsample,2))
        else:
            high = 1000
            
            self.data = np.random.randint(1, high=high, size=(100,10))
        self.label = np.ones((nsample,2))
        
        #Normalize
        self.data = (self.data - self.data.mean())/(high - 1)
        
        
        for i in range(len(self.data)):
            for j in range(len(self.data[i])-1):
                if self.data[i][j+1] < self.data[i][j]:
                    self.label[i][1] = 0
                    break
            else:
                self.label[i][0] = 0
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index],self.label[index][:]


# In[76]:


class cls_classifier(nn.Module):
    def __init__(self):
        super(cls_classifier, self).__init__()
        self.classifier = nn.Sequential(
            
            nn.Linear(10,50)
            ,nn.ReLU()
            ,nn.Linear(50,2)
            ,nn.Softmax(dim=1)
            )
    def forward(self,seq):
        logits = self.classifier(seq)
        return logits


# In[77]:


def get_accuracy_from_logits(logits, labels):
    
    y_predicted = torch.argmax(logits, dim=1)
    y_actual = torch.argmax(labels, dim=1)
    
    acc = float(1.0 - np.count_nonzero(y_actual - y_predicted)/len(y_actual))*100
    
    
    return acc


# In[78]:


x_train = cls_data(nsample=5000,is_train=True)
x_val = cls_data(nsample=100,is_train=False)

loader_train = DataLoader(x_train, batch_size=8, num_workers=0)
loader_val = DataLoader(x_val, batch_size=len(x_val), num_workers=0)

net = cls_classifier()
net.to(torch.device("cpu"))
criterion = nn.BCEWithLogitsLoss()
opti = torch.optim.Adam(net.parameters(),lr=0.001)
for ep in range(50):

    for idx,(seq,labels) in enumerate(loader_train):
        
        opti.zero_grad()
        
        logits = net(seq.float())
        
        loss = criterion(logits.squeeze(-1), labels.float())
        
        loss.backward()
        
        opti.step()
        
        if idx%100 == 0:

            acc = get_accuracy_from_logits(logits, labels)        
            print(acc)
    
    logits = net(torch.Tensor(loader_val.dataset.data))
    acc = get_accuracy_from_logits(logits, torch.Tensor(loader_val.dataset.label))        
    print(acc)


# In[40]:


logits = net(torch.Tensor(loader_val.dataset.data))
acc = get_accuracy_from_logits(logits, torch.Tensor(loader_val.dataset.label))        
print(acc)


# In[93]:


a = torch.Tensor([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,30],[1,2,3,4,5,60,7,8,9,30]])
b = (a - torch.mean(a)) / (torch.max(a)-1)
net.forward(b)

