import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as trn_models
import numpy as np

# Beam prediction model relying on input beam sequences alone
class RecNet(nn.Module):
    def __init__(self,
                 inp_dim,
                 hid_dim,
                 out_dim,
                 out_seq,
                 num_layers,
                 drop_prob=0.2):
        super(RecNet, self).__init__()
        self.hid_dim = hid_dim
        self.out_seq = out_seq
        # self.orig_dim = orig_dim
        self.num_layers = num_layers

        # Define layers
        self.gru = nn.GRU(inp_dim,hid_dim,num_layers,batch_first=True,dropout=drop_prob)
        self.classifier = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)--> This is commented out because softmax is implemented in the definition of cross entropy (Line 34 in model_train.py)

    def forward(self,x,h):
        out, h = self.gru(x,h)
        out = self.relu(out[:,-1*self.out_seq:,:])
        y = self.classifier(out)
        # y = self.softmax(out)
        return [y, h]

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim) )


