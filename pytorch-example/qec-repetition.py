#rewrite the layers input

import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch

# config
L=5
trials=300000
LAYERS=[L-1,L*8,L*8*4,L*4,L]
n_epochs = 250 #250   # number of epochs to run
batch_size = 64*100 #10  # size of each batch

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



#from data import *
#X,y,_a,_b = load_data(title='FashionMNIST')
#X,y = X.to(device),y.to(device)


# generate qec data
def repetition(L):
    H=torch.zeros((L-1,L),dtype=torch.int8)
    for i in range(L-1):
        H[i,i]=1
        H[i,i+1]=1
    return H

def generate_data(L,trials):
    H = repetition(L)
    print(H)
    p=0.2
    e = (torch.rand((trials,L)) + p).floor()
    e=e.type(torch.int8)
    s = e @ torch.t(H) % 2
    X = s
    y=e
    return X,y


X,y=generate_data(L,trials)
X_test,y_test = generate_data(L,trials//10)

#from data import *
#save_data(X,y,X_test,y_test,f'repetition{L}')
#print(X)
#print(y)
#exit()

X = X.type(torch.float32)
y = y.type(torch.float32)
X_test = X_test.type(torch.float32)
y_test = y_test.type(torch.float32)
print('X.shape,y.shape',X.shape,y.shape)
#t,l=X.shape
#X=X.reshape([trials, 1, L-1])
#y=y.reshape([trials,1,L])
#print(X.shape,y.shape)

#input('...')

class Deep(nn.Module):
    def __init__(self,layers=[28*28,640,640,60,10]):
        super().__init__()
        #self.flatten = nn.Flatten() #add for mnist
        modules=[]
        print('processing layers:',layers)
        num_layers=len(layers)
        for i in range(num_layers-2):
            layer0 = layers[i]
            layer1 = layers[i+1]
            layer = nn.Linear(layer0,layer1)
            act = nn.ReLU()
            modules.append(layer)
            modules.append(act)
        self.linear_relu_stack = nn.Sequential(*modules)
        self.output = nn.Linear(layers[-2], layers[-1])
        
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        
    def forward(self, x):
        #x = self.flatten(x) #add for mnist
        x = self.linear_relu_stack(x)
        x = self.output(x)
        x = self.sigmoid(x)
        #x = self.softmax(x)
        return x

def acc_eval(y_pred,y_batch):
    #return ((y_pred>0) == y_batch).type(torch.float).mean()
    return ((y_pred.round()) == y_batch).type(torch.float).mean()
    
def model_train(model, X_train, y_train, X_val, y_val):
    for i in [X_train, y_train, X_val, y_val]:
        print(i.shape)
    # loss function and optimizer
    #loss_fn = nn.BCELoss()  # binary cross entropy
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #n_epochs = 100 #250   # number of epochs to run
    #batch_size = 64*10 #10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        # print(epoch)
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                X_batch,y_batch = X_batch.to(device),y_batch.to(device)
                #print('X_batch.shape',X_batch.shape)
                # forward pass

                
                y_pred = model(X_batch)
                #print(y_pred.shape,y_batch.shape)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress                
                #acc = ((y_pred>0) == y_batch).type(torch.float).mean()
                acc = acc_eval(y_pred,y_batch)
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val=X_val.to(device)
        y_val=y_val.to(device)        
        y_pred = model(X_val)
                #acc = ((y_pred>0) == y_val).type(torch.float).mean()
        acc = acc_eval(y_pred,y_val)        
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    #skip best acc
    #return acc
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

#from sklearn.model_selection import StratifiedKFold, train_test_split

cv_scores = []
#model = Deep().to(device)
#for train, test in kfold.split(X,y[:,1]):
if True:
    # create model, train, and get accuracy
    layers=LAYERS    
    model = Deep(layers).to(device)
    print(model)
    #acc = model_train(model, X[train], y[train], X[test], y[test])
    acc = model_train(model, X, y, X_test, y_test)
    print("Accuracy (wide): %.2f" % acc)
    cv_scores.append(acc.cpu())
    #break
    
# evaluate the model
#cv_scores=cv_scores.cpu()
acc = np.mean(cv_scores)
std = np.std(cv_scores)
print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))
