import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as trn_models
import numpy as np

class modResNet18(nn.Module):
    def __init__(self,
                 inp_dim,
                 out_dim):
        super(modResNet18, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # Prepare image-descriptor extractor
        e = trn_models.resnet18(pretrained=True)
        layers = list(e.children())[:-2]
        assert np.mod(inp_dim[1],32) == 0 and np.mod(inp_dim[2],32) == 0, 'Input dimensions do not work'
        self.kernel_dim = ( int( inp_dim[1]/32 ), int( inp_dim[2]/32 ) )
        linear_proj = nn.Conv2d(512,self.out_dim,self.kernel_dim)
        layers.append(linear_proj)
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class DeepVBT(nn.Module):
    def __init__(self,
                 encoder,
                 num_layers,
                 inp_dim,
                 hid_dim,
                 out_dim,
                 orig_dim,
                 drop_prob):
        super(DeepVBT, self).__init__()
        self.encoder = encoder
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.orig_dim = orig_dim
        self.num_layers = num_layers

        # Define layers
        self.gru = nn.GRU(inp_dim, hid_dim, num_layers, batch_first=False, dropout=drop_prob)
        self.classifier = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim) )

    def forward(self, x,y,h):
        num_seq = x.shape[1]
        d = torch.zeros((x.shape[1],x.shape[0],self.inp_dim//2)).cuda()
        for i in range(num_seq):
            d[i,:,:] = torch.squeeze( self.encoder( torch.squeeze(x[:,i,:,:,:]) ) )
        y = y.permute(1,0,2)
        X = torch.cat([d,y],dim=2)
        out, h = self.gru(X, h)
        out = self.relu(out[-1, :, :])
        out = self.classifier(out)
        return [out, h]

# Beam prediction model relying on input beam sequences alone
class RecNet(nn.Module):
    def __init__(self,
                 inp_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 drop_prob=0.2):
        super(RecNet, self).__init__()
        self.hid_dim = hid_dim
        # self.orig_dim = orig_dim
        self.num_layers = num_layers

        # Define layers
        self.gru = nn.GRU(inp_dim,hid_dim,num_layers,batch_first=True,dropout=drop_prob)
        self.classifier = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self,x,h):
        out, h = self.gru(x,h)
        out = self.relu(out[:,-1:,:])
        y = self.classifier(out)
        # y = self.softmax(out)
        return [y, h]

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim) )


