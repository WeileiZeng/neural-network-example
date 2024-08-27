print('''
Author: Weilei Zeng
Date: 2024-08-27
# copied from exp-regression.py
  simulate a step function

idea: this can be used for error correction. First use a classification for logical codeword, then one only need to fix the syndrome of the pure error!
Or one can add a post processing. Given error e0, a decoder suggest a fix e1, based on syndrome s and e1, NN claim whenther a logical error has occured
''')
'''
# statistics
use same structure, i.e. n=10
epoch 8069, training loss 0.07825344055891037,         validation loss 0.07710275799036026


#
The classification is not working, get same class for all input. Can we train a classification model in advance. Or one has to use label data?
'''


# ELu give e-5 loss with single layer of size 8
# ReLU give 5e-3 loss with much larger n network


import torch
import torch.nn as nn

from torch.autograd import Variable

device = 'cuda:7'
#torch.cuda.set_device(0)
torch.set_default_device(device)

#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

def get_data(n):
    n = int(n)
    X = torch.rand(n,1)*2
    y = X.floor() - X + 1
    return X,y
def data_test():
    X,y = get_data(10)
    print(X,y)    

#data_test()
#exit()


x_test,y_test=get_data(1e3)
print('mean x y',x_test.mean(),y_test.mean())
#x_test=torch.rand((int(1e5),1))*10 + 1
#y_test = func(x_test)

class PreClassificationModel(torch.nn.Module):

        def __init__(self):                
                super(PreClassificationModel, self).__init__()
                l=64
                out=10
                self.classification = nn.Sequential(
                    nn.Linear(1,l),
                    nn.BatchNorm1d(l), nn.ReLU(), nn.Dropout(),nn.Linear(l,l),
                    nn.BatchNorm1d(l), nn.ReLU(), nn.Dropout(),nn.Linear(l,l),
                    nn.BatchNorm1d(l), nn.ReLU(), nn.Dropout(),nn.Linear(l,l),
                    nn.BatchNorm1d(l), nn.ReLU(), nn.Dropout(),nn.Linear(l,l),
                    nn.BatchNorm1d(l), nn.ReLU(), nn.Dropout(),
                    nn.Linear(l,out),
                    #nn.BatchNorm1d(10), nn.ReLU(), nn.Dropout(),nn.Linear(10,10),
                    #nn.Dropout(), nn.ReLU(), nn.Linear(10,10),
                    nn.Softmax(),
                    #nn.Sigmoid()
                )
                self.parallel = [ nn.Linear(1,1) for _ in range(10) ]
                
                
        def forward(self, x):
                b = self.classification(x)
                #print(b.shape)
                li = [p(x) for p in self.parallel]
                _ = torch.stack(li, dim=0)
                _ = _.squeeze()                
                #_ = torch.vstack(li)
                
                #_ = torch.cat( **( p(x) for p in self.parallel ))
                #print(_.shape)
                y = b @ _
                y = y.diagonal()
                y = y.reshape((y.shape[0],1))
                #print(y.shape)
                return y

def test_model(model):
    # test the classification of the model
    #x = torch.Tensor([[1.0], [2.0], [3.0]])
    x = torch.range(1,10).reshape((10,1)) -0.3
    #tensor([[0.7000], [1.7000], [2.7000], [3.7000], [4.7000], [5.7000], [6.7000], [7.7000], [8.7000], [9.7000]])
    x = x.to(device)
    b = model.classification(x)
    print(b)
    values, indices = b.max(dim=1)
    #print(values)
    print(indices)
    
our_model = PreClassificationModel()
print(our_model)

tt
#test_model(our_model)
#exit()

#criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.0000001)
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.0001)

for epoch in range(500000):
        our_model.train()
        x_data,y_data = get_data(16)
        #x_data=torch.rand((int(1e3),1))*10 + 1
        #y_data = func(x_data)
        
        # Forward pass: Compute predicted y by passing 
        # x to the model
        pred_y = our_model(x_data)

        # Compute and print loss
        loss = criterion(pred_y, y_data)

        # Zero gradients, perform a backward pass, 
        # and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(' epoch {}, loss {}'.format(epoch, loss.item()), end='\r' )

        if epoch % 1000 == 0:
                our_model.eval()
                pred_y = our_model(x_test)
                loss_val = criterion(pred_y, y_test)
                #print(pred_y[:10])
                #print(y_test[:10])                
                print(' '*35,' epoch {}, training loss {},        \tvalidation loss {}.    '.format(epoch, loss.item(), loss_val.item()), end='\r' )
                #input()
                test_model(our_model)
        
print('here')

#new_var = Variable(torch.Tensor([[4.0]]))
#pred_y = our_model(new_var)
#print("predict (after training)", 4, our_model(new_var).item())
