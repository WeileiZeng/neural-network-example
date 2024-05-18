print('''
to generate the lattice of size 3 x 3 , with 1,...,9 filled in each blank, and each row, col, diagnal add up to 15
pinn, no training data, just use the right loss function

input: 3 x 3 tensor with 1 as the beginning, or 5 in the center
output:
prob: 3x3x9 tensor, the 9-tensor contain prob for entry being 1,...,9 respectively
logits:argmax(9-tensor)

modified code from reference
https://towardsdatascience.com/solving-differential-equations-with-neural-networks-4c6aa7b31c51
''')

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

# config
batch_size = 64 #64*1 #10  # size of each batch
# Number of epochs
num_epochs = int(1e6)
HIDDEN_SIZE=3*3*9 *9*9
torch.set_printoptions(8)


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNet(nn.Module):
    def __init__(self, hidden_size, output_size=1,input_size=1):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.LeakyReLU()
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.LeakyReLU()
        self.l5 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

        
    def forward(self, x):
        out = self.flatten(x)
        out = self.l1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.relu4(out)
        out = self.l5(out)
        out = self.softmax(out)
        return out


import random
#return random x for input
def get_x():
    a = list(range(1,10))
    random.shuffle(a)
    #print(a)
    a = torch.tensor(a,dtype=torch.float)
    a=a.to(device).reshape((1,3,3))
    #print(a)
    return a

#x = torch.tensor( list(range(1,10))).float()
x=torch.zeros((1,3,3),dtype=torch.float)
#x[0,1,1]=1.
x = x.reshape((1,3,3))
x = x.to(device)
x.requires_grad_(True)

x_base = x
print('x',x)

# Constant for the model
#k = 1

# Instantiate one model with 50 neurons on the hidden layers
model = NeuralNet(hidden_size=HIDDEN_SIZE,input_size=9,output_size=3*3*9).to(device)

# Loss and optimizer
learning_rate = 8e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters())

#H = repetition(L)
#Hf = H.float()
#m = nn.Sigmoid()


b1 = torch.ones((9,9,9),dtype=torch.float) *  bench.reshape((9,1,1))
b2 = torch.ones((9,9,9),dtype=torch.float) *  bench.reshape((1,9,1))
b3 = torch.ones((9,9,9),dtype=torch.float) *  bench.reshape((1,1,9))
b = b1 +b2 +b3
b = b.to(device) #include all terms of  i+j+k:  b[i,j,k]=i+j+k

# return expectation value of the sum (i+j+k) from the three lists
# each lists contains probs for being 1,...,9
# each input is a 1D tensor of length 9
# this sum over i,j,k for prob(i)*prob(j)*prob(k)*(i+j+k)
def expected_sum(v1,v2,v3):
    #SAMPLE INPUT FOR A COLUMN: v1,v2,v3 = y_pred[0,0],y_pred[1,0],y_pred[2,0]
    v1 = v1.reshape((9,1,1))
    v2 = v2.reshape((1,9,1))
    v3 = v3.reshape((1,1,9))
    prob = (v2 @ v1) @v3 #contain prob for i j k
    r = torch.einsum('ijk,ijk->',prob,b).to(device)
    return r

bench = torch.tensor(list(range(1,10))).float()

softmax2 = nn.Softmax(dim=2)
softmax0 = nn.Softmax(dim=0)
mse = nn.MSELoss()
for epoch in range(num_epochs):


    # Forward pass
    x = x_base + get_x()
    y_pred = model(x)

    #define loss
    y_pred = softmax0(y_pred.reshape((9,9))) #softmax for given value among all locations
    y_pred = y_pred.reshape((3,3,9))
    #y_pred = softmax2(y_pred)  #softmax for all values in each blank

    #recalculate the expectation value
    r = [expected_sum(*y_pred[:,_]) for _ in range(3)] #cols
    c = [expected_sum(*y_pred[_,:]) for _ in range(3)] #rows
    diagnol1 = expected_sum(y_pred[0,0] , y_pred[1,1] , y_pred[2,2])
    diagnol2 = expected_sum(y_pred[0,2] , y_pred[1,1] , y_pred[2,0])
    r.extend(c)
    r.append(diagnol1)
    r.append(diagnol2)
    expect = torch.tensor(r) # contain all expectation values, that should give 15
    
    
    base = torch.ones_like(expect) * 15.
    loss1 = mse(expect,base) #make sure all expectation value matches 15
    
    
    #expectation=(bench @ y_pred) .sum(2)


    #loss 2: exclusive
    #elements in each black should be exclusive to other blanks
    r = y_pred.prod(0).prod(0)
    loss2 = r.sum()*1.e7

    loss = loss1 + loss2
    logits = y_pred.argmax(2)+1


    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('y_pred[0,0]',y_pred[0,0])
        print('expect',expect)
        print(logits)
        print(f'epoch {epoch}:,loss={loss:.5f},loss1={loss1},loss2={loss2}')
        #input()
        #print(f'epoch {epoch}: acc_s={acc:.3f},acc_e = {acc_e:.3f},loss={loss:.5f},{loss1},{loss2}')
        #print(epoch,loss, 'acc',acc,'loss',loss1,loss2)
        #print(epoch,loss, 'acc',acc)
    
