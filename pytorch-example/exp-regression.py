print('''
Example of linear regression using pytorch
code from
https://www.geeksforgeeks.org/linear-regression-using-pytorch/
''')


import torch
from torch.autograd import Variable

#device = 'cuda'
#torch.cuda.set_device(0)
torch.set_default_device('cuda:0')

#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# config
c1=0.01 #0.01
c2=10.
c3=10.
def func(x):
        y = c2*torch.exp(c1*x) + c3
        return y


def gen(n): # e.g. n=1e5        
        x = torch.rand((int(n),1))*10 + 1
        y = func(x)
        return x,y
x_test,y_test=gen(1e5)
print('mean x y',x_test.mean(),y_test.mean())
#x_test=torch.rand((int(1e5),1))*10 + 1
#y_test = func(x_test)

class LinearRegressionModel(torch.nn.Module):

        def __init__(self):
                super(LinearRegressionModel, self).__init__()
                self.linear = torch.nn.Linear(1, 1) # One in and one out
                L=64*4*8
                self.sequential = torch.nn.Sequential(
                        torch.nn.Linear(1, L),
                        
                        torch.nn.ELU(L),torch.nn.Linear(L, L),
                        torch.nn.ELU(L),torch.nn.Linear(L, L),
                        #torch.nn.ReLU(L),
                        #torch.nn.Linear(L, L),
                        #torch.nn.ReLU(L),
                        #torch.nn.Linear(L, L),

                        torch.nn.Linear(L, 1),
                        )
                
        def forward(self, x):
                #y_pred = self.linear(x)
                y_pred = self.sequential(x)
                return y_pred

# our model
our_model = LinearRegressionModel()

#criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.0000001)
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.0000001)

for epoch in range(5000000):

        our_model.train()
        x_data,y_data = gen(1e3)
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
        
print()
new_var = Variable(torch.Tensor([[4.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())
