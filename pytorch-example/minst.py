import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm

import torch


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


#import pandas as pd
'''
# Read data
data = pd.read_csv("content/sonar.all-data", header=None)
#data[61]=data[60]
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

print('data size',data.shape)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)



#import torch
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

print(y.shape)
y=torch.cat((y,y),1)
print(y.shape)
'''

from data import *
X,y,_a,_b = load_data(title='FashionMNIST')

#X,y = X.to(device),y.to(device)

X = X.type(torch.float32)
#y = y.type(torch.float32)
print(X.shape,y.shape)
X=X.reshape([60000, 1, 28*28])
#y.reshape([60000, 1, 1])
print(X.shape,y.shape)

#input('...')

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #add for mnist
        self.layer1 = nn.Linear(28*28, 640)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(640, 640)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(640, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 10)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x) #add for mnist
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = (self.output(x))
        #x = self.sigmoid(self.output(x))
        return x


def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    #loss_fn = nn.BCELoss()  # binary cross entropy
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 100 #250   # number of epochs to run
    batch_size = 64*1*16 #10  # size of each batch
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
                #correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
                #acc = (y_pred.argmax(1) == y_batch).type(torch.float).sum().item()
                acc = (y_pred.argmax(1) == y_batch).type(torch.float).mean()
                #print(y_pred)
                #print(y_pred.argmax(1))
                #print(y_batch)
                #print(acc)
                #input(...)
                #acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val=X_val.to(device)
        y_val=y_val.to(device)        
        y_pred = model(X_val)
        #acc = (y_pred.round() == y_val).float().mean()
        #acc = (y_pred.argmax(1) == y_val).type(torch.float).sum().item()
        acc = (y_pred.argmax(1) == y_val).type(torch.float).mean()
        #acc = float(acc)
        #acc = (y_pred.argmax(1) == y_batch).type(torch.float).mean()
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

from sklearn.model_selection import StratifiedKFold, train_test_split

# train-test split: Hold out the test set for final model evaluation
# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []
for train, test in kfold.split(X,y):
    # create model, train, and get accuracy
    print('train',train)
    print('test',test)
    print(X[train].shape)
    model = Deep().to(device)
    print(model)
    acc = model_train(model, X[train], y[train], X[test], y[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores.append(acc.cpu())
    break

# evaluate the model
#cv_scores=cv_scores.cpu()
acc = np.mean(cv_scores)
std = np.std(cv_scores)
print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))
