# Code from mostly from https://medium.com/@ramamurthi96/a-simple-neural-network-model-for-mnist-using-pytorch-4b8b148ecbdc
# numpy==1.21.4

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize
from skimage import io, transform
import os
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
 
INPUT  = '/pfs/mnist'
OUTPUT = '/pfs/out'
 
class CustomMNISTDataset(Dataset):
    def __init__(self, name, transform=None):
        if name != 'testing' and name != 'training':
            raise AttributeError('Name must be testing or training.')
        self.transform = transform
        self.images = []
 
        for subdir in os.listdir(os.path.join(INPUT, 'mnist_png', name)):
            for file in os.listdir(os.path.join(INPUT, 'mnist_png', name, subdir)):
                self.images.append((subdir, os.path.join(INPUT, 'mnist_png', name, subdir, file)))
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        (label, path) = self.images[idx]
        image = io.imread(path)
 
        # Reshaping the array from 1*784 to 28*28
        image = image.reshape(28,28)
        #image = image.astype(float)
 
        # Scaling the image so that the values only range between 0 and 1
        image = image/255.0
       
        if self.transform:
            image = self.transform(image)
   
        #image = image.to(torch.float)   
        
        return image.to(torch.float32), int(label)
 
class MyOwnNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyOwnNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()
        )
 
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return  logits
 
def train(dataloader, model, loss_fn, optimizer):
    # Total size of dataset for reference
    size = 0
   
    # places your model into training mode
    model.train()
   
    # loss batch
    batch_loss = {}
    batch_accuracy = {}
   
    correct = 0
    _correct = 0
   
    # Gives X , y for each batch
    for batch, (X, y) in enumerate(dataloader):
        # Converting device to cuda
        X, y = X.to(device), y.to(device)
        model.to(device)
       
        # Compute prediction error / loss
        # 1. Compute y_pred
        # 2. Compute loss between y and y_pred using selectd loss function
       
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
 
        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
        _batch_size = len(X)
       
        correct += _correct
       
        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size
       
        size += _batch_size
       
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")
   
    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
   
    return batch_loss , batch_accuracy
 
def validation(dataloader, model, loss_fn):
    # Total size of dataset for reference
    size = 0
    num_batches = len(dataloader)
   
    # Setting the model under evaluation mode.
    model.eval()
 
    test_loss, correct = 0, 0
   
    _correct = 0
    _batch_size = 0
   
    batch_loss = {}
    batch_accuracy = {}
   
    with torch.no_grad():
        # Gives X , y for each batch
        for batch , (X, y) in enumerate(dataloader):
           
            X, y = X.to(device), y.to(device)
            model.to(device)
            pred = model(X)
           
            batch_loss[batch] = loss_fn(pred, y).item()
            test_loss += batch_loss[batch]
            _batch_size = len(X)
           
            _correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct
           
            size+=_batch_size
            batch_accuracy[batch] = _correct/_batch_size
   
    ## Calculating loss based on loss function defined
    test_loss /= num_batches
   
    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size
   
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
   
    return batch_loss , batch_accuracy
 
if __name__ == '__main__':
    for subdir in sorted(os.listdir(os.path.join(INPUT, 'mnist_png/training'))):
        print('{0}\t{1}'.format(subdir, len(os.listdir(os.path.join(INPUT, 'mnist_png/training', subdir)))))
 
    test_ds = CustomMNISTDataset('testing')
 
    transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,)),])
    train_ds = CustomMNISTDataset('training', transform=transforms)
 
    training_indices = list(range(train_ds.__len__()))
    train_indices, test_indices = train_test_split(training_indices, test_size=0.1)
 
    #x0, y0 = train_ds[0]
    #print(x0.shape, y0)
 
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)
 
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, sampler=train_sampler, num_workers=16)
    valid_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, sampler=valid_sampler, num_workers=16)
 
    #x0, y0 = next(iter(train_dataloader))
    #print(x0.shape)
    #print(y0.shape)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = MyOwnNeuralNetwork().to(device)
    print(model)
 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
 
    train_batch_loss = []
    train_batch_accuracy = []
    valid_batch_accuracy = []
    valid_batch_loss = []
    train_epoch_no = []
    valid_epoch_no = []
 
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_dataloader, model, loss_fn, optimizer)
        _valid_batch_loss , _valid_batch_accuracy = validation(valid_dataloader, model, loss_fn)
        for i in range(len(_train_batch_loss)):
            train_batch_loss.append(_train_batch_loss[i])
            train_batch_accuracy.append(_train_batch_accuracy[i])
            train_epoch_no.append( t + float((i+1)/len(_train_batch_loss)))    
        for i in range(len(_valid_batch_loss)):
            valid_batch_loss.append(_valid_batch_loss[i])
            valid_batch_accuracy.append(_valid_batch_accuracy[i])
            valid_epoch_no.append( t + float((i+1)/len(_valid_batch_loss)))    
    print("Done!")
 
    figure = plt.figure(figsize=(16, 16))
 
    figure.add_subplot(2, 2, 1)
    plt.plot(train_epoch_no , train_batch_accuracy)
    plt.title("Train Batch Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Train Accuracy")
 
    figure.add_subplot(2, 2, 2)
    plt.plot(train_epoch_no , train_batch_loss)
    plt.title("Train Batch Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
 
    figure.add_subplot(2, 2, 3)
    plt.plot(valid_epoch_no , valid_batch_accuracy)
    plt.title("Valid Batch Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Train Accuracy")
 
    figure.add_subplot(2, 2, 4)
    plt.plot(valid_epoch_no , valid_batch_loss)
    plt.title("Valid Batch Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
 
    plt.savefig(os.path.join(OUTPUT, 'figure.png'), bbox_inches='tight')
    torch.save(model.state_dict(), os.path.join(OUTPUT, 'mnist.pt'))
