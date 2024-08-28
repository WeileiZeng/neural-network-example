print('''
Author: Weilei Zeng
Date: 2024-08-27
# train a classification model to simulate a floor function
next step: use two model for every other region
''')
note='''
# statistics
For width =0.98
 epoch 27200, training loss 1.4710922241210938,           validation loss 1.509655475616455.      acc 0.9539999961853027
tensor([1, 1, 2, 3, 5, 5, 6, 8, 8, 9, 1, 2, 2, 4, 5, 5, 7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9], device='cuda:7')
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32) reference

For width = 0.8
epoch 22800, training loss 1.524492859840393,            validation loss 1.4781208038330078.     acc 0.9850000143051147
tensor([1, 1, 3, 4, 4, 5, 6, 8, 8, 9, 1, 2, 3, 4, 5, 6, 6, 8, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9], device='cuda:7')
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32) reference

For width =0.6
epoch 25200, training loss 1.5184835195541382,           validation loss 1.4661011695861816.     acc 0.9980000257492065
tensor([0, 2, 3, 4, 4, 6, 7, 8, 9, 9, 1, 2, 3, 4, 4, 6, 7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9], device='cuda:7')
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32) reference

For width = 0.4
epoch 8000, training loss 1.4835686683654785,            validation loss 1.4741820096969604.     acc 1.0
tensor([0, 2, 2, 3, 5, 6, 6, 8, 9, 9, 1, 2, 3, 4, 5, 6, 6, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9], device='cuda:7')
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32) reference

For every other region
acc=1

'''

import torch
import torch.nn as nn
#from torch.autograd import Variable

# CONFIG
batch_size = 16
test_size = int(1e6)
n_epoches = int(1e5)
hidden_size = 10
num_hidden_layers = 2
pieces = 10 # number of pieces in the piecewise functions to be simulated
width=0.8


device = 'cuda:7'
#torch.cuda.set_device(0)
torch.set_default_device(device)
torch.set_printoptions(linewidth=150)


def get_every_other_data(n):
    # only return data in every other range of length 1
    n = int(n)
    ii = torch.randint(pieces//2,(n,1))
    ff = torch.rand(n,1)
    X = ii*2 + ff
    col_indices = ii.squeeze()
    y = torch.zeros(n,pieces)
    row_indices = torch.range(0,y.shape[0]-1, dtype=torch.long)
    y[row_indices,col_indices]=1
    return X,y
def get_short_data(n):
    # only return data in range ~ width
    #width=0.9
    n = int(n)
    ii = torch.randint(pieces,(n,1))
    ff = torch.rand(n,1)*width + (1-width)/2   # change data range to be a short smooth region
    X = ii+ff
    col_indices = ii.squeeze()
    y = torch.zeros(n,pieces)
    row_indices = torch.range(0,y.shape[0]-1, dtype=torch.long)
    y[row_indices,col_indices]=1
    return X,y


def get_data(n):
    return get_every_other_data(n)
    return get_short_data(n)
    n = int(n)
    X = torch.rand(n,1) * pieces
    col_indices = X.floor().int().squeeze()
    #y = X.floor() - X + 1
    y = torch.zeros(n,pieces)
    #print(y)
    row_indices = torch.range(0,y.shape[0]-1, dtype=torch.long)
    #print(row_indices)
    #print(col_indices)
    y[row_indices,col_indices]=1

    return X,y

def data_test():
    X,y = get_data(5)
    print('sample data X,y of length 5')
    print(X)
    print(y)    

data_test()
#exit()
#tt

x_test,y_test=get_data(1e3)
print('mean x y',x_test.mean(),y_test.mean())
#x_test=torch.rand((int(1e5),1))*10 + 1
#y_test = func(x_test)

class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block
    def forward(self, x):
        return x + self.block(x)

class PreClassificationModel(torch.nn.Module):

        def __init__(self):                
                super(PreClassificationModel, self).__init__()
                l=hidden_size
                ll = hidden_size * 10
                out=10
                act=nn.Sigmoid
                def get_block(ll):
                      module_ = nn.Sequential(
                            #nn.BatchNorm1d(ll), 
                            act(), 
                            #nn.Dropout(),
                            nn.Linear(ll,ll))
                      res_=ResBlock(module_)
                      return res_
                relu_stack = [get_block(ll)]*num_hidden_layers

                #relu_stack=[nn.BatchNorm1d(ll), act(), nn.Dropout(),nn.Linear(ll,ll)]*8
                self.classification = nn.Sequential(
                    nn.Linear(1,ll),
                    #nn.BatchNorm1d(l), act(), nn.Dropout(),nn.Linear(l,ll),
                    *relu_stack,
                    #nn.BatchNorm1d(ll), nn.ReLU(), nn.Dropout(),nn.Linear(ll,ll),
                    #nn.BatchNorm1d(ll), nn.ReLU(), nn.Dropout(),nn.Linear(ll,ll),
                    nn.BatchNorm1d(ll), act(), nn.Dropout(),nn.Linear(ll,out),
                    #nn.BatchNorm1d(l), act(), nn.Dropout(),
                    #nn.Linear(l,out),
                    #nn.BatchNorm1d(10), nn.ReLU(), nn.Dropout(),nn.Linear(10,10),
                    #nn.Dropout(), nn.ReLU(), nn.Linear(10,10),
                    nn.Softmax(),
                    #nn.Sigmoid()
                )
                #self.parallel = [ nn.Linear(1,1) for _ in range(10) ]
                
                
        def forward(self, x):
                b = self.classification(x)
                return b

def test_model(model):
    # test the classification of the model
    #x = torch.Tensor([[1.0], [2.0], [3.0]])
    x0 = torch.range(1,10).reshape((10,1))
    x = torch.cat([x0+0.01,x0+0.1,x0+0.3,x0+0.5])

    #x = tensor([[0.7000], [1.7000], [2.7000], [3.7000], [4.7000], [5.7000], [6.7000], [7.7000], [8.7000], [9.7000]])
    x = x.to(device)
    b = model.classification(x)
    #print(b)
    eps=1e-5
    _ = b * (b>eps)
    #print(_)
    values, indices = b.max(dim=1)
    #print(values)
    print(indices) # expect to be [0,1,2,3,4,5,6,7,8,9]
    r0=torch.range(0,pieces-1).type(torch.int)
    r = torch.cat([r0]*4)
    print(r,'reference')
    
our_model = PreClassificationModel()
print(our_model)


#test_model(our_model)
#exit()

#criterion = torch.nn.MSELoss(size_average = False)
#criterion = torch.nn.MSELoss()
#criterion = torch.nn.BCELoss()
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.BCEWithLogitsLoss()
#optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.0000001)
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)

def accuracy(pred_y,y_data):
      #print(pred_y)
      #print(y_data)
      y1=torch.argmax(pred_y,dim=1)
      y2=torch.argmax(y_data,dim=1)
      #print(y1)
      #print(y2)
      #tt
      #exit()
      rate = 1- torch.count_nonzero(y1-y2)/pred_y.shape[0]
      return rate


for epoch in range(500000):
        our_model.train()
        x_data,y_data = get_data(batch_size)
        #y_data = torch.max(y_data, -1)[1]
        #y_data = y_data.flatten().type(torch.LongTensor).to(device)
        #y_data = y_data.squeeze()
        #x_data = x_data.double()
        #y_data = y_data.type(torch.LongTensor).to(device)
        # Forward pass: Compute predicted y by passing 
        # x to the model
        pred_y = our_model(x_data)
        #.argmax(dim=1)

        #print(pred_y)
        #print(y_data)
        # Compute and print loss
        loss = criterion(pred_y, y_data)

        # Zero gradients, perform a backward pass, 
        # and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(' epoch {}, loss {}'.format(epoch, loss.item()), end='\r' )

        if epoch % 200 == 0:
                our_model.eval()
                pred_y = our_model(x_test)
                loss_val = criterion(pred_y, y_test)
                acc = accuracy(pred_y,y_test)
                #print(pred_y[:10])
                #print(y_test[:10])                
                print(' '*5,' epoch {}, training loss {},        \tvalidation loss {}.    \tacc {}'.format(epoch, loss.item(), loss_val.item(), acc.item()), end='\n' )
                #end='\r'
                #input()
                test_model(our_model)
        
print('here')

#new_var = Variable(torch.Tensor([[4.0]]))
#pred_y = our_model(new_var)
#print("predict (after training)", 4, our_model(new_var).item())
