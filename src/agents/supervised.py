import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128, 64)  # 6*6 from image dimension
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x            

def loadTrainingData(path):
    training = []

    trainingFile = open(path, "r")

    for ram in trainingFile:
        data = [int(numeric_string) for numeric_string in ram.split(' ')[:-1]]
        training.append(torch.tensor(data, dtype=torch.uint8))
    
    return training

def loadTestingData(path):
    testing = []

    testingFile = open(path, "r")

    for num in testingFile.readline().split(' ')[:-1]:
        data = int(num)
        testing.append(torch.tensor(data, dtype=torch.uint8))
    
    return testing
    

#../res/training data/ram.txt

def train():

    net = Net()
    net = net.float()

    trainingData = loadTrainingData('../res/training data/ram.txt')
    testingData = loadTestingData('../res/training data/action.txt')

    for i in range(0, 10):
        print(trainingData[i])
        print(testingData[i])

train()







