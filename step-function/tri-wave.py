print('''
Author: Weilei Zeng
Date: 2024-08-27
# copied from exp-regression.py
  simulate a step function
''')
'''
# statistics
use same structure, i.e. n=10
epoch 8069, training loss 0.07825344055891037,         validation loss 0.07710275799036026
'''


# ELu give e-5 loss with single layer of size 8
# ReLU give 5e-3 loss with much larger n network


import torch
import torch.nn as nn

from torch.autograd import Variable

#device = 'cuda'
#torch.cuda.set_device(0)
torch.set_default_device('cuda:4')

#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

def get_data(n):
    n = int(n)
    X = torch.rand(n,1)*10
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
                self.classification = nn.Sequential(
                        nn.Linear(1,10),
                        nn.Sigmoid()
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


our_model = PreClassificationModel()
print(our_model)


#criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.0000001)
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)

for epoch in range(500000):
        our_model.train()
        x_data,y_data = get_data(1e3)
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

        if epoch % 1 == 0:
                our_model.eval()
                pred_y = our_model(x_test)
                loss_val = criterion(pred_y, y_test)
                #print(pred_y[:10])
                #print(y_test[:10])                
                print(' epoch {}, training loss {},        \tvalidation loss {}.    '.format(epoch, loss.item(), loss_val.item()), end='\r' )
                #input()
        
print('here')

#new_var = Variable(torch.Tensor([[4.0]]))
#pred_y = our_model(new_var)
#print("predict (after training)", 4, our_model(new_var).item())
