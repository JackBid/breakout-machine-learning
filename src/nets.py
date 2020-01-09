import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FC364(nn.Module):

    def __init__(self):
        super(FC364, self).__init__()
        self.fc1 = nn.Linear(3, 6) 


        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
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
