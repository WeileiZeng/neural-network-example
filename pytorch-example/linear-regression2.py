print('''
Example of linear regression using pytorch
code from
https://www.geeksforgeeks.org/linear-regression-using-pytorch/
''')


import torch
from torch.autograd import Variable

#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

x_data=torch.rand((int(1e6),1))*10 + 1
y_data = x_data * 2 +1

class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

# our model
our_model = LinearRegressionModel()

#criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.001)

for epoch in range(5000000):

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
	print(' epoch {}, loss {}'.format(epoch, loss.item()), end='\r' )
        
print()
new_var = Variable(torch.Tensor([[4.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())
