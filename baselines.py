import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        
        kernel_length = 61
        in_chans = 61
        input_time_length = 100
        F1=args.F1 #8
        D=args.D #2
        self.F2 = F1*D
        drop_prob=args.p_dropout #0.25

        pool_class =nn.AvgPool2d
        self.model = nn.Sequential()
        self.model.add_module(
            "conv_temporal",
            nn.Conv2d(
                1,
                F1,
                (1, kernel_length),
                stride=1,
                bias=False,
                padding=(0, kernel_length // 2),
            ),
        )
        self.model.add_module(
                    "bnorm_temporal",
                    nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
                )
        self.model.add_module(
            "conv_spatial",
            Conv2dWithConstraint(
                F1,
                F1 * D,
                (in_chans, 1),
                max_norm=1,
                stride=1,
                bias=False,
                groups=F1,
                padding=(0, 0),
            ),
        )
        self.model.add_module(
            "bnorm_1",
            nn.BatchNorm2d(
                F1 * D, momentum=0.01, affine=True, eps=1e-3
            ),
        )
        self.model.add_module("elu_1", nn.ELU())
        self.model.add_module(
            "pool_1", pool_class(kernel_size=(1, 4), stride=(1, 4))
        )
        self.model.add_module("drop_1", nn.Dropout(p=drop_prob))
        self.model.add_module(
            "conv_separable_depth",
            nn.Conv2d(
                F1 * D,
                F1 * D,
                (1, 16),
                stride=1,
                bias=False,
                groups=F1 * D,
                padding=(0, 16 // 2),
            ),
        )
        self.model.add_module(
            "conv_separable_point",
            nn.Conv2d(
                F1 * D,
                F1 * D,
                (1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            ),
        )
        self.model.add_module(
            "bnorm_2",
            nn.BatchNorm2d(F1*D, momentum=0.01, affine=True, eps=1e-3),
        )
        self.model.add_module("elu_2", nn.ELU())
        self.model.add_module(
            "pool_2", pool_class(kernel_size=(1, 8), stride=(1, 8))
        )
        self.model.add_module("drop_2", nn.Dropout(p=drop_prob))
        
        self.fc = nn.Linear(self.F2*(100//32), 3)        

    def forward(self, x):
        # Layer 1
        x = self.model(x).view(-1,self.F2*(100//32))
        if self.training:
            return torch.sigmoid(self.fc(x)), torch.zeros(10).to(x.device)
        else:
            return torch.sigmoid(self.fc(x))
        
    def __str__(self):
        return "EEGNet"
    
class shGCN(torch.nn.Module):
    def __init__(self, args):
        super(shGCN, self).__init__()
        conv_nonlin = lambda x: x*x
        pool_nonlin = lambda x: torch.log(torch.clamp(x, min=1e-6))
        self.encoder = nn.Conv1d(
                1,
                args.d_latent,
                args.kernel_size,
                stride=1,
                padding=args.kernel_size//2
            )
        self.fc1 = nn.Linear(100,1)
        self.processor = gnn.GraphConv(args.d_latent, args.d_hidden)
        self.bnorm = nn.BatchNorm1d(
                args.d_hidden, momentum=0.1, affine=True
            )
        self.conv_nonlin = Expression(conv_nonlin)
        self.aggregator = globPool(args.aggregate)
        self.pool_nonlin = Expression(pool_nonlin)
        self.drop = nn.Dropout(p=args.p_dropout)
        self.fc2 = nn.Linear(args.d_hidden, 3)

    def forward(self, x, edge_index, edge_weight = None, batch = None):
        x = self.encoder(x.unsqueeze(1))
        x = self.fc1(x).squeeze()
        x = self.processor(x, edge_index)
        x = self.bnorm(x)
        x = self.conv_nonlin(x)
        x, grads = self.aggregator(x, batch)
        x = self.pool_nonlin(x)
        x = self.drop(x)
        y = self.fc2(x)
        if self.training:
            return torch.sigmoid(y), grads
        else:
            return torch.sigmoid(y)
    
class Expression(torch.nn.Module):

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)
    
class ShallowNet(nn.Module):
    def __init__(self, args):
        super(ShallowNet, self).__init__()
        
        in_chans = 61
        input_time_length=100
        
        n_filters_time=args.n_filters_time #40
        self.n_filters_spat=args.n_filters_spat #40
        filter_time_length=args.filter_time_length #25
        pool_time_length=args.pool_time_length #75
        pool_time_stride=args.pool_time_stride #15
        drop_prob=args.p_dropout #0.5
        batch_norm_alpha=0.1
        
        self.out_dim = int(self.n_filters_spat*np.ceil(((input_time_length-filter_time_length+1) - pool_time_length + 1)/pool_time_stride))
        
        conv_nonlin = lambda x: x*x
        pool_nonlin = lambda x: torch.log(torch.clamp(x, min=1e-6))
        pool_class = nn.AvgPool2d
        
        self.model = nn.Sequential()
        self.model.add_module(
            "conv_time",
            nn.Conv2d(
                1,
                n_filters_time,
                (filter_time_length, 1),
                stride=1,
            ),
        )
        self.model.add_module(
            "conv_spat",
            nn.Conv2d(
                n_filters_time,
                self.n_filters_spat,
                (1, in_chans),
                stride=1,
                bias=False,
            ),
        )
        self.model.add_module(
            "bnorm",
            nn.BatchNorm2d(
                self.n_filters_spat, momentum=batch_norm_alpha, affine=True
            ),
        )
        self.model.add_module("conv_nonlin", Expression(conv_nonlin))
        self.model.add_module(
            "pool",
            pool_class(
                kernel_size=(pool_time_length, 1),
                stride=(pool_time_stride, 1),
            ),
        )
        self.model.add_module("pool_nonlin", Expression(pool_nonlin))
        self.model.add_module("drop", nn.Dropout(p=drop_prob))

        self.fc = nn.Linear(self.out_dim, 3)
        
    def forward(self, x):
        x = self.model(x).view(-1, self.out_dim)
        if self.training:
            return torch.sigmoid(self.fc(x)), torch.zeros(10).to(x.device)
        else:
            return torch.sigmoid(self.fc(x))
    
    def __str__(self):
        return "ShallowConvNet"
    
class Potential(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(2*in_dim, in_dim)
        self.lnorm1 = LayerNorm(in_dim)
        #self.res1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim, in_dim//2)
        self.lnorm2 = LayerNorm(in_dim//2)
        self.fc3 = nn.Linear(in_dim//2, 10)
        self.res3 = nn.Linear(in_dim//2, 10)
       
    def forward(self, x):
        h = self.lnorm1(F.tanh(self.fc1(x))) 
        #x = self.res1(x)
        #h += x
        h = self.lnorm2(F.tanh(self.fc2(h)))
        h = self.fc3(h)
        #x = self.res3(x)
        #h += x
        return h
        
class EquilibriumAgg(torch.nn.Module):
    def __init__(self, in_dim, t = 10, alpha = 1e-1, reg_weight = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.t = t
        self.alpha = alpha
        self.reg_weight = reg_weight
        self.potential = Potential(in_dim)
        
    def forward(self, x, batch):
        bsize = int(batch.max().item() + 1)
        grads = torch.zeros(self.t, bsize, self.in_dim).to(x.device)
        y = torch.zeros(bsize, self.in_dim, requires_grad=True).to(x.device)
        for i in range(self.t):
            u = self.energy(x, y, bsize)
            grad = self.grad(u, y)
            y = y - self.alpha*grad
            grads[i] = grad
        return y, grads    
            
    def grad(self, u, y):
        grads = torch.ones(u.shape, device=u.device)
        grad = torch.autograd.grad(
                            u,
                            y,
                            create_graph=True,
                            grad_outputs=grads)[0]
        return grad
             
    def energy(self, x, y, bsize):
        residual = self.residual(y)
        x = x.view(bsize, -1, self.in_dim)
        n_nodes = x.shape[1]
        y = y.repeat(1, n_nodes).view(bsize, -1, self.in_dim)
        inputs = torch.cat([x,y], -1)
        return self.potential(inputs).square().mean(dim = -1).sum(-1) + residual
        
    def residual(self, y):
        return self.reg_weight*y.square().mean(-1)
