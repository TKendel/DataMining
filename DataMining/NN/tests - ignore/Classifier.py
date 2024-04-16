#!/usr/bin/python3
# This code presents an implementation of LeNet-5
"""
This code is an implementation of the Convolutional Neural Network LeNet-5, as presented by Y. LeCun et al. in 1998 in 'Gradient Based Learning Applied to Do cument Recognition'. The code is a current adaptation of the model utilizing PyTorch, and it was completed following the documentation foud in:

S. Chintala, “Deep learning with Pytorch: A 60 minute blitz”, PyTorch Tutorials Documentation. [Online]. Available: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html. [Accessed: 04-May-2021].


The PyTorch Foundation, "Neuronal Networks", PyTorch Tutorials Documentation. [Online]. Available: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html. [Accessed: 02-May-2021].


The PyTorch Foundation, "PyTorch Documnentation", PyTorch Tutorials Documentation. [Online]. Available: https://pytorch.org/docs/stable/index.html. [Accessed: 02-May-2021].


E. Lewinson, “Implementing Yann LeCun's LeNet-5 in PyTorch,” GitHub, 09-May-2020. [Online]. Available: https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb. [Accessed: 04-May-2021]. 

Supervised by professor Justin Vasselli
CS 4256: Machine Learning course, Spring 2021, Bennington College
"""


import sys

import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt




# Check device for compatible cuda GPU (nVidia) [Recommended]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 12   #Absolute number of training epochs
N_CLASSES = 10
DOWNLOAD = False    #Default value

# Request if dowloading the MIND dataset is necesary
dwld_mnist = str(input("Would you like to dowload the MNIST dataset? Y/N"+"\n"))
if dwld_mnist.lower() == "y": #If the answeer is anthing but Y/y, we assume the dowload is not needed.
    DOWNLOAD = True


        
#Training function
def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



    #Validation
def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss




def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # Temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # Change the plot style to default
    plt.style.use('default')





# Training Loop
def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)





# Define transforms
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# Download and create datasets                              #********* Change data set used******
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=DOWNLOAD)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# Define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)




#LeNet-5 implementation (architecture)
class ClassicLeNet5(nn.Module):

    def __init__(self, n_classes):
        super(ClassicLeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs





"""
Although this second implementation is repetitive, as it mostly utilizes all of the same features from the first model, it was left like this for reference when studying the achitechture.
"""

#LeNet-5 implementation with Maxpooling Subsampling layer and ReLU activation layer
class AltLeNet5(nn.Module):

    def __init__(self, n_classes):
        super(AltLeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(                                     #********* Probably won't use feature extraction******
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(), # Updated to --> nn.ReLU(), from nn.Tanh(),
            nn.MaxPool2d(kernel_size=2), # Updated to --> nn.MaxPool2d(), from nn.AvgPool2d()
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(), # Updated to --> nn.ReLU(), from nn.Tanh(), from nn.AvgPool2d()
            nn.MaxPool2d(kernel_size=2), # Updated to --> nn.MaxPool2d(), from nn.AvgPool2d()
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU() # Updated to --> nn.ReLU() from nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):                                                           #********* Probably won't use feature extraction******
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class Classification(nn.Module):

    def __init__(self, n_classes):
        super(AltLeNet5, self).__init__()
        
        """
        self.feature_extractor = nn.Sequential(                                     #********* Probably won't use feature extraction******
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(), # Updated to --> nn.ReLU(), from nn.Tanh(),
            nn.MaxPool2d(kernel_size=2), # Updated to --> nn.MaxPool2d(), from nn.AvgPool2d()
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(), # Updated to --> nn.ReLU(), from nn.Tanh(), from nn.AvgPool2d()
            nn.MaxPool2d(kernel_size=2), # Updated to --> nn.MaxPool2d(), from nn.AvgPool2d()
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU() # Updated to --> nn.ReLU() from nn.Tanh()
        )
        """

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=56),
            nn.ReLU(),
            nn.Linear(in_features=56, out_features=n_classes),
        )


    def forward(self, x):                                                           #********* Probably won't use feature extraction******
        #x = self.feature_extractor(x)
        #x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    



MODEL = ClassicLeNet5   # default to Classic LeNt-5
if "alt" in sys.argv:   # If Alt is in the arguments, select th AltLeNet5, however, if too many arguments are given, the code will prevent it from running in the following section.
    MODEL = AltLeNet5

# Message if neither argument 'classic' or 'alt' is found
message = """To run a classic LeNet-5 architecture, please run the command:
    'python3 LeNet-5.py classic'
To run a LeNet-5 implementation with Maxpooling Subsampling layer and ReLU activation layer, please run the command:
    'python3 LeNet-5.py alt'
    """

if "alt" not in sys.argv and "classic" not in sys.argv: #If neither Classic or Alt arguments were imputed, request them.
    print(message)
elif "alt" in sys.argv and "classic" in sys.argv: #If both Classic or Alt arguments were imputed, report so and do not run the training.
    print("Too many arguments were found.")
else:   #Run training
    torch.manual_seed(RANDOM_SEED)


    model = MODEL(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)


        # Save the model
    file_name = str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')+' model.pth')
    torch.save(model.state_dict(), file_name)
    print("Saved PyTorch Model State to", file_name)