
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models


class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.flatten = torch.nn.Flatten()
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        x = self.flatten(x)
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)

class Googlnet_modified(nn.Module):
    def __init__(self):
        super().__init__()
        self.google = models.googlenet(pretrained=True)
   
        self.flatten= nn.Flatten()
        self. Linear1= nn.Linear(1000,150)
        self.relu =nn.ReLU()
        self.Linear2= nn.Linear(150,2)
        self.relu2= nn.ReLU()
    def forward(self, input):
        google_out= self.google(input)
        out= self.flatten(google_out)
        out = self.Linear1(out)
        out = self.relu(out)
        out= self.Linear2(out)
        out = self.relu2(out)
        return F.log_softmax(out, dim=1)

class Custom_Net(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.feature_extract= nn.Sequential(
            nn.Conv2d(3, 1, (1, 1), stride=1), # 1x 1 convolution first
            nn.Conv2d(1, 10, (3, 3), stride=1),
            nn.BatchNorm2d(10),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 50, (3, 3), stride=1),
            nn.BatchNorm2d(50),
            nn.MaxPool2d((3, 3), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 250, (3, 3), stride=1),
            nn.BatchNorm2d(250),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten())
        self.nodes = (((((((image_size-2)//2)-2)//2)-2)//2)**2)*250
        self.linear1 =nn.Linear (self.nodes, 1000)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(1000,150)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(150,2)
        self.relu3= nn.ReLU()
    def forward(self, input):
        features= self.feature_extract(input)
        out= self.linear1(features)
        out= self.relu(out)
        out= self.linear2(out)
        out=self.relu2(out)
        out= self.linear3(out)
        out= self.relu3(out)
        return F.log_softmax(out, dim=1)
