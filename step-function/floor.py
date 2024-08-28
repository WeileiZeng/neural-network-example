print('''
Author: Weilei Zeng
Date: 2024-08-27
# train a classification model to simulate a floor function
''')
note='''
# statistics

'''

import torch
import torch.nn as nn
#from torch.autograd import Variable

# CONFIG
batch_size = 16
test_size = 1000
n_epoches = int(1e5)
hidden_size = 10
pieces = 10 # number of pieces in the piecewise functions to be simulated

device = 'cuda:7'
#torch.cuda.set_device(0)
torch.set_default_device(device)
torch.set_printoptions(linewidth=150)

def get_short_data(n):
    # only return data in range 0.25-0.75
    n = int(n)
    ii = torch.randint(pieces,(n,1))
    ff = torch.rand(n,1)*0.2+0.4   # change data range to be a short smooth region
    X = ii+ff
    col_indices = ii.squeeze()
    y = torch.zeros(n,pieces)
    row_indices = torch.range(0,y.shape[0]-1, dtype=torch.long)
    y[row_indices,col_indices]=1
    return X,y


def get_data(n):
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
                relu_stack = [get_block(ll)]*8

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
    x = torch.range(1,10).reshape((10,1)) - 0.45
    #x = tensor([[0.7000], [1.7000], [2.7000], [3.7000], [4.7000], [5.7000], [6.7000], [7.7000], [8.7000], [9.7000]])
    x = x.to(device)
    b = model.classification(x)
    #print(b)
    eps=1e-5
    print( b * (b>eps))
    values, indices = b.max(dim=1)
    #print(values)
    print(indices) # expect to be [0,1,2,3,4,5,6,7,8,9]
    print(torch.range(0,pieces-1).type(torch.int),'reference')
    
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
                #print(pred_y[:10])
                #print(y_test[:10])                
                print(' '*35,' epoch {}, training loss {},        \tvalidation loss {}.    '.format(epoch, loss.item(), loss_val.item()), end='\n' )
                #end='\r'
                #input()
                test_model(our_model)
        
print('here')

#new_var = Variable(torch.Tensor([[4.0]]))
#pred_y = our_model(new_var)
#print("predict (after training)", 4, our_model(new_var).item())
