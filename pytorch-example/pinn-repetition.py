print('''
PINN, solve differential equation
https://towardsdatascience.com/solving-differential-equations-with-neural-networks-4c6aa7b31c51
''')

import numpy as np
import torch
import torch.nn as nn


# config
L=9
trials=300000
#LAYERS=[L-1,L*8,L*8*4,L*4,L]
#n_epochs = 250 #250   # number of epochs to run
batch_size = 64 #64*1 #10  # size of each batch
# Number of epochs
num_epochs = int(1e7)

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

# Create the criterion that will be used for the DE part of the loss
criterion = nn.MSELoss()

# Define the loss function for the initial condition
def initial_condition_loss(y, target_value):
    return nn.MSELoss()(y, target_value)

# In this model, do not provide any data, just follow the coriterin
# try to minimize diff between s and s(e)
# s -> e, check s(e) vs s

#there should be two loss term. one to match syndrome. one to minize weight
def syndrome_match_loss(s,e,H):
    se = e.type(torch.int8)@H.t() % 2
    acc = 1- ((s + se) % 2 ).float().mean()
    #print('-'*50)
    #print(s)
    #print(se)
    #print((s + se) % 2)
    #input()
    return acc, nn.BCELoss()(s,se.float())
#return acc,nn.MSELoss()(s.float(),se.float())


def min_weight_loss(e):
    #print(e)
    return e.float().mean()/10.
            
# Time vector that will be used as input of our NN
t_numpy = np.arange(0, 5+0.01, 0.01, dtype=np.float32)
t = torch.from_numpy(t_numpy).reshape(len(t_numpy), 1)
t.requires_grad_(True)

# my input is the syndrome
def generate_batch(batch_size,H):
    #def generate_data(L,trials):
    #H = repetition(L)
    #print(H)
    #p=0.2
    #when H is full rank ,syndrome could be any random vector. if H not full rank, then syndrome can only be images of H
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
model = NeuralNet(hidden_size=50,input_size=L-1,output_size=L)

# Loss and optimizer
learning_rate = 8e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



H = repetition(L)
Hf = H.float()

m = nn.Sigmoid()

s = torch.rand((batch_size,L-1),requires_grad=True).round()
x_train_base = torch.zeros((batch_size,L-1),requires_grad=True)

for epoch in range(num_epochs):

    # Randomly perturbing the training points to have a wider range of times
    #epsilon = torch.normal(0,0.1, size=(len(t),1)).float()
    #t_train = t + epsilon
    #t.requires_grad_(True)
    #input('...')
    #x_trial = torch.rand((batch_size,L-1),requires_grad=True).round()

#def generate_data(L,trials):
    #H = repetition(L)
    #print(H)
    p=0.1
    e0 = (torch.rand((batch_size,L)) + p).floor()
    e0i=e0.type(torch.int8)
    s = e0i @ torch.t(H) % 2
    #X = s
    #y=e
    #return X,y


    
    #s = torch.rand((batch_size,L-1),requires_grad=True).round()    
    #s.requires_grad_(True)
    x_train = x_train_base + s.float()
    
    # Forward pass
    #y_pred = model(t_train)
    y_pred = model(x_train)

    #print('y_pred', y_pred)    
    e=y_pred.round()  #do I need sigmoid?

    #print(e)
    #print('e',e)
    #input('...')
    #calculate syndrome
    #se = e @ H %2

    se = e.type(torch.int8)@H.t() % 2
    acc = 1- ((s + se) % 2 ).float().mean() #check if syndrome matches!

    #check if error matches
    acc_e = 1-((e - e0i )%2).float().mean() 

    
    se2 = y_pred@Hf.t() % 2
    loss1 = nn.MSELoss()(se2,x_train)
    # loss 1 minimize the difference to the syndrome
    
    #loss1 = nn.BCELoss()(m(s),se.float())
    #loss2 = e.float().mean()/10.
    #return acc, nn.BCELoss()(s,se.float())

    
    #acc,loss1 = syndrome_match_loss(s,e,H)    
    #loss2 = min_weight_loss(e)
    #acc,loss1 = syndrome_match_loss(s,e,H)    
    #loss2 = min_weight_loss(e)
    loss2 = y_pred.mean()/10.
    loss = loss1 + loss2

    #loss = loss1
    
    '''
    # Calculate the derivative of the forward pass w.r.t. the input (t)
    dy_dt = torch.autograd.grad(y_pred, 
                                t_train, 
                                grad_outputs=torch.ones_like(y_pred), 
                                create_graph=True)[0]

    # Define the differential equation and calculate the loss
    loss_DE = criterion(dy_dt + k*y_pred, torch.zeros_like(dy_dt))

    # Define the initial condition loss
    loss_IC = initial_condition_loss(model(torch.tensor([[0.0]])), 
                                     torch.tensor([[1.0]]))

    loss = loss_DE + loss_IC
    '''
    
    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f'epoch {epoch}: acc_s={acc:.3f},acc_e = {acc_e:.3f},loss={loss:.5f},{loss1},{loss2}')
        #print(epoch,loss, 'acc',acc,'loss',loss1,loss2)
        #print(epoch,loss, 'acc',acc)
    
