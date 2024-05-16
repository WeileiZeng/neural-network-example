print('''
train neural network to decoder repetition code without data
only use the parity check to verify the right syndrome
in the meantime, minimize weight of output error
the network learn to do the job

modified code from reference
https://towardsdatascience.com/solving-differential-equations-with-neural-networks-4c6aa7b31c51
''')

import numpy as np
import torch
import torch.nn as nn


# config
L=79
#trials=300000
#LAYERS=[L-1,L*8,L*8*4,L*4,L]
#n_epochs = 250 #250   # number of epochs to run
batch_size = 64 #64*1 #10  # size of each batch
# Number of epochs
num_epochs = int(1e7)
HIDDEN_SIZE=L*6

# generate qec data
def repetition(L):
    H=torch.zeros((L-1,L),dtype=torch.int8)
    for i in range(L-1):
        H[i,i]=1
        H[i,i+1]=1
    return H



class NeuralNet(nn.Module):
    def __init__(self, hidden_size, output_size=1,input_size=1):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.LeakyReLU()
        self.l4 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        #perhaps need a sigmoid
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.sigmoid(out)
        return out


# In this model, do not provide any data, just follow the coriterin
# try to minimize diff between s and s(e)
# s -> e, check s(e) vs s


            
# Time vector that will be used as input of our NN
t_numpy = np.arange(0, 5+0.01, 0.01, dtype=np.float32)
t = torch.from_numpy(t_numpy).reshape(len(t_numpy), 1)
t.requires_grad_(True)

# my input is the syndrome
def generate_batch(batch_size,H):
    s = torch.rand((batch_size,L)).round()
    return s
    e = (torch.rand((trials,L)) + p).floor()
    e=e.type(torch.int8)
    s = e @ torch.t(H) % 2
    X = s
    y=e
    return X,y


# Constant for the model
k = 1

# Instantiate one model with 50 neurons on the hidden layers
model = NeuralNet(hidden_size=HIDDEN_SIZE,input_size=L-1,output_size=L)

# Loss and optimizer
learning_rate = 8e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



H = repetition(L)
Hf = H.float()

m = nn.Sigmoid()


x_train_base = torch.zeros((batch_size,L-1),requires_grad=True)

for epoch in range(num_epochs):


    p=0.1
    e0 = (torch.rand((batch_size,L)) + p).floor()
    e0i=e0.type(torch.int8)
    s = e0i @ torch.t(H) % 2
    #s = torch.rand((batch_size,L-1),requires_grad=True).round()    #generate syndrome directly
    #s.requires_grad_(True)
    x_train = x_train_base + s.float()
    
    # Forward pass
    y_pred = model(x_train)
    e=y_pred.round()

    se = e.type(torch.int8)@H.t() % 2  #calculate binary syndrome of output error
    acc = 1- ((s + se) % 2 ).float().mean() #check if syndrome matches!

    #check if error matches
    acc_e = 1-((e - e0i )%2).float().mean() 
    
    se2 = y_pred@Hf.t() % 2  #float version of syndrome to estimate lose
    loss1 = nn.MSELoss()(se2,x_train)    # loss 1 minimize the difference to the syndrome
    #note: use float numbers to loss estimation
    
    loss2 = y_pred.mean()/10.  #minimize weight of output error. use float version!
    loss = loss1 + loss2

    
    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'epoch {epoch}: acc_s={acc:.3f},acc_e = {acc_e:.3f},loss={loss:.5f},{loss1},{loss2}')
        #print(epoch,loss, 'acc',acc,'loss',loss1,loss2)
        #print(epoch,loss, 'acc',acc)
    
