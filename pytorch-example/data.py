print("""
load and save data in a single file according to title
""")


import torch
from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor

#save data
#title='FashionMNIST'

class DataClass(nn.Module):
    def __init__(self):#,X_train,y_train,X_test,y_test,title='tmp'):
        super(DataClass, self).__init__()
    def set(self,X_train,y_train,X_test,y_test,title='tmp'):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.title=title
        
def save_data(X_train,y_train,X_test,y_test,title='tmp'):
    #data=DataClass()
    #data.set(X_train,y_train,X_test,y_test,title)
    data=dict(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
    filename='data-'+title+'.pt'    
    torch.save(data,filename)
    print(f'data saved to {filename} with {str(data.keys())}')

def load_data(title='tmp'):
    #model = TheModelClass(*args, **kwargs)
    filename='data-'+title+'.pt'
    #data=DataClass(*args, **kwargs)
    data=torch.load(filename)
    print(f'read {filename} with {str(data.keys())}')
    return data['X_train'],data['y_train'],data['X_test'],data['y_test']
    

def test():
    #create sample data, save it, and then load it
    pass

if __name__=="__main__":
    test()
