import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class StraightLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha  = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
    def forward(self , x):
        return self.alpha + self.beta * x
class Y(nn.Module):
    def __init__(self):
        super().__init__()
        self.f  = StraightLine()
        self.sigma = nn.Parameter(torch.tensor(1.0))
    def forward(self , x, y):
        pred = self.f(x)
        return -0.5*torch.log(2*np.pi*self.sigma**2+(y-pred)**2/2/self.sigma**2)
    
model=Y()
optimizer = optim.Adam(model.parameters())
#epoch = 0
#from collections.abc import Iterable
#with Interruptable() as check_interrupted:
#with Iterable() as check_interrupted:
#    check_interrupted()
#x = 13.375
#y=3.457
n=100000
x= np.array(list(range(n)))
np.random.shuffle(x)
y=x * 12.345678901234 + 91.23456789012345
x=torch.tensor(x)
y=torch.tensor(y)

torch.set_printoptions(precision=16)
#batch_size=100
for epoch in range(n*100):
    optimizer.zero_grad()
    i = epoch % n
    loglik = model(x[i:i+10], y[i:i+10])
    #print(loglik ,(x[i], y[i]))
    e = - torch.mean(loglik)
    e.backward()
    optimizer.step()
    #IPython.display.clear_output(wait=True)
    
    #print(model)
    if epoch % 10000 ==0:
        print('alpha',model.f.alpha)
        print('beta',model.f.beta)
        print('sigma',model.sigma)
        print(f'epoch={epoch} loglik={-e.item():.3}')
        #input('...')
#    epoch += 1
