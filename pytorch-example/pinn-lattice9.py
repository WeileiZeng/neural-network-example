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
#L=9
#trials=300000
#LAYERS=[L-1,L*8,L*8*4,L*4,L]
#n_epochs = 250 #250   # number of epochs to run
batch_size = 64 #64*1 #10  # size of each batch
# Number of epochs
num_epochs = int(1e6)
HIDDEN_SIZE=3*3*9 *9*3

torch.set_printoptions(8)


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#device='cpu'
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
        self.l4 = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        #perhaps need a sigmoid
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.l1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.softmax(out)
        #out = self.sigmoid(out)
        return out


# In this model, do not provide any data, just follow the coriterin
# try to minimize diff between s and s(e)
# s -> e, check s(e) vs s


            
# Time vector that will be used as input of our NN
#t_numpy = np.arange(0, 5+0.01, 0.01, dtype=np.float32)
#t = torch.from_numpy(t_numpy).reshape(len(t_numpy), 1)
#t.requires_grad_(True)

x = torch.tensor( list(range(1,10))).float()
#x=torch.zeros((1,3,3),dtype=torch.float)
#x[0,1,1]=1.
x = x.reshape((1,3,3))
x = x.to(device)
x.requires_grad_(True)

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


# return expectation value of the sum of the three lists
# each lists contains probs for being 1,...,9
# each input is 1 1D tensor of length 9
def expected_sum(v1,v2,v3):
    #v1,v2,v3 = y_pred[0,0],y_pred[1,0],y_pred[2,0]
    v1 = v1.reshape((9,1,1))
    v2 = v2.reshape((1,9,1))
    v3 = v3.reshape((1,1,9))
    prob = (v2 @ v1) @v3 #contain prob for i j k

    #print(v2 @ v1)
    #exit()
        
    b1 = torch.ones((9,9,9),dtype=torch.float) *  bench.reshape((9,1,1))
    #print(b1)
    #exit()
    b2 = torch.ones((9,9,9),dtype=torch.float) *  bench.reshape((1,9,1))
    b3 = torch.ones((9,9,9),dtype=torch.float) *  bench.reshape((1,1,9))
    b = b1 +b2 +b3
    b = b.to(device)
    r = torch.einsum('ijk,ijk->',prob,b).to(device)
    return r


#x_train_base = torch.zeros((batch_size,L-1),requires_grad=True)
bench = torch.tensor(list(range(1,10))).float()

softmax2 = nn.Softmax(dim=2)
softmax0 = nn.Softmax(dim=0)
mse = nn.MSELoss()
for epoch in range(num_epochs):


    #p=0.1
    #e0 = (torch.rand((batch_size,L)) + p).floor()
    #e0i=e0.type(torch.int8)
    #s = e0i @ torch.t(H) % 2
    #s = torch.rand((batch_size,L-1),requires_grad=True).round()    #generate syndrome directly
    #s.requires_grad_(True)
    #x_train = x_train_base + s.float()
    
    # Forward pass
    #y_pred = model(x_train)
    y_pred = model(x)
    #print('raw output\n',y_pred)
    #exit()
    #e=y_pred.round()

    #se = e.type(torch.int8)@H.t() % 2  #calculate binary syndrome of output error
    #acc = 1- ((s + se) % 2 ).float().mean() #check if syndrome matches!

    #check if error matches
    #acc_e = 1-((e - e0i )%2).float().mean() 
    
    #se2 = y_pred@Hf.t() % 2  #float version of syndrome to estimate lose
    #loss1 = nn.MSELoss()(se2,x_train)    # loss 1 minimize the difference to the syndrome
    #note: use float numbers to loss estimation
    
    #loss2 = y_pred.mean()/10.  #minimize weight of output error. use float version!
    #loss = loss1 + loss2


    #define loss
    # copy the tensor and then reshape. Not sure if this affect the calculation of grad in y_pred
    #_ = torch.empty_like(y_pred)
    #_.copy_(y_pred)
    #y_pred = _
    #y_pred =
    y_pred = softmax0(y_pred.reshape((9,9)))
    y_pred = y_pred.reshape((3,3,9))
    #y_pred = softmax2(y_pred)
    #print('adter normalization\n',y_pred)
    #exit()
    #logits = y_pred.argmax(1)
    #acc
    #loss 1 add up to 15
    #get expectation value for each blank


    #recalculate the expectation value
    #expected_sum(v1,v2,v3):
    r = [expected_sum(*y_pred[:,_]) for _ in range(3)]
    c = [expected_sum(*y_pred[_,:]) for _ in range(3)]
    #print(r,c)
    diagnol1 = expected_sum(y_pred[0,0] , y_pred[1,1] , y_pred[2,2])
    diagnol2 = expected_sum(y_pred[0,2] , y_pred[1,1] , y_pred[2,0])
    r.extend(c)
    r.append(diagnol1)
    r.append(diagnol2)
    #print(r)
    expect = torch.tensor(r)
    #for each col
    #exit()
    #print('a',a)
    #exit()
    
    
    #print('bench',bench)
    #get expectation value for each blank
    #print(y_pred)
    #expectation = torch.einsum('ijk,k->ij',[y_pred,bench])
    #print('expectation',expectation)
    #rows = expectation.sum(1)
    #cols = expectation.sum(0)
    #diagnol1 = torch.tensor([expectation[0,0] + expectation[1,1] + expectation[2,2]])
    #diagnol2 = torch.tensor([expectation[0,2] + expectation[1,1] + expectation[2,0]])
    #print(rows,cols,diagnol1,diagnol2)
    #expect = torch.cat((rows,cols,diagnol1,diagnol2))
    base = torch.ones_like(expect) * 15.
    #print(expect,base)
    loss1 = mse(expect,base)
    #print(loss1)
    
    
    #expectation=(bench @ y_pred) .sum(2)


    #loss 2: exclusive: minimize prod(y_pred[:,:,i])
    #loss_row = 

    #loss2  = y_pred.prod(2).sum() + y_pred.prod(0).sum()+y_pred.prod(1).sum()
    #need diagnol term

    #diagnol1 = y_pred[0,0]

    #elements in each black should be exclusive to other blanks
    r = y_pred.prod(0).prod(0)
    loss2 = r.sum()*1.e7
    #print(r)
    #print(y_pred.prod(0))
    #print(y_pred.prod(0).sum())

    #exit()
    loss = loss1 + loss2

    #loss = loss2

    logits = y_pred.argmax(2)+1

    
    #input()
    #y_pred.reshape((1,81))
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
    
