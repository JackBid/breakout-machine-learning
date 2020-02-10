import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FullyConnected(nn.Module):

    def __init__(self, inputLayer=3, hiddenLayer=4):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(inputLayer, hiddenLayer) 
        self.fc2 = nn.Linear(hiddenLayer, 4)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class DeepFullyConnected(nn.Module):

    def __init__(self, inputLayer, hiddenLayer1, hiddenLayer2):
        super(DeepFullyConnected, self).__init__()
        self.fc1 = nn.Linear(inputLayer, hiddenLayer1) 
        self.fc2 = nn.Linear(hiddenLayer1, hiddenLayer2)
        self.fc3 = nn.Linear(hiddenLayer2, 4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class Test(nn.Module):

    def __init__(self):
        super(Test, self).__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 6) 
        self.fc3 = nn.Linear(6, 4)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

