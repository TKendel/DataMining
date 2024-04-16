#!/usr/bin/python3


import sys

import numpy as np
from datetime import datetime   #For saving files with timestamp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# Check device for compatible cuda GPU (nVidia) [Recommended]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Length of 22 --> 23 - 1 becaue we are not including ID as factor







# Parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 6   #Absolute number of training epochs
N_CLASSES = 10     # mood, numbers 1 through 10


N_LAYERS = 2    #***Layers of the RNN
S_INPUT = 28 #22     #Number of features, size of input #******
S_SEQUENCE = 28 #1
N_HIDDEN = 128 #64 #hidden_size = 64






DOWNLOAD = False    #Default value  #Download datazset

# Request if dowloading the MIND dataset is necesary
dwld_mnist = str(input("Would you like to dowload the MNIST dataset? Y/N"+"\n"))
if dwld_mnist.lower() == "y": #If the answeer is anthing but Y/y, we assume the dowload is not needed.
    DOWNLOAD = True




class LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_layers, num_class) :
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn. Linear(hidden_size, num_class)

    def forward (self, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out



class RNN(nn.Module):   
    #https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/

    #https://medium.com/@reddyyashu20/rnn-python-code-in-keras-and-pytorch-6ab842a85e15

    def __init__(self, input_size, hidden_size, n_layers, n_classes):
        #super(RNN, self).__init__()
        #self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        #self.fc = nn.Linear(hidden_size, n_classes)


        #https://medium.com/@reddyyashu20/rnn-python-code-in-keras-and-pytorch-6ab842a85e15
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, n_classes)

        #THis would be for a simpler RNN (non-LSTM)
        #https://www.youtube.com/watch?v=Gl2WXLIMvKA
        """
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)  # batch_first=True when providing as (batch, seq, features)
        """

    def forward(self, x):                                                           #********* Probably won't use feature extraction******
        
        """
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
        """
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output[-1])
        return output, hidden







        
#Training function
def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    #FROM NEW****+*+*+**+*
    """
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    """
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








 




MODEL = LSTM   # default to Classic LeNt-5
#****
"""
if "alt" in sys.argv:   # If Alt is in the arguments, select th AltLeNet5, however, if too many arguments are given, the code will prevent it from running in the following section.
    MODEL = AltLeNet5
"""



# Message if neither argument 'classic' or 'alt' is found
message = """To run a classic LeNet-5 architecture, please run the command:
    'python3 LeNet-5.py classic'
To run a LeNet-5 implementation with Maxpooling Subsampling layer and ReLU activation layer, please run the command:
    'python3 LeNet-5.py alt'
    """
"""
if "alt" not in sys.argv and "classic" not in sys.argv: #If neither Classic or Alt arguments were imputed, request them.
    print(message)
elif "alt" in sys.argv and "classic" in sys.argv: #If both Classic or Alt arguments were imputed, report so and do not run the training.
    print("Too many arguments were found.")
else:   #Run training
    torch.manual_seed(RANDOM_SEED)
"""
if True:

    
    model = MODEL(S_INPUT, N_HIDDEN, N_LAYERS, N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #criterion = nn.CrossEntropyLoss()  #***Using the other


    #From NEW **+*+*+*+
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()                         #<-- This one is different
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #From NEW **+*+*+*+


    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)


        # Save the model
    file_name = str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')+' model.pth')
    torch.save(model.state_dict(), file_name)
    print("Saved PyTorch Model State to", file_name)