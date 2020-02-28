import numpy as np
import gym
import torch
import math
import time
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Util():
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net.cuda()
        else:
            self.device = torch.device('cpu')

    # Convert an array to string
    def arrToString(self, arr):
        arrString = ''
        for val in arr:
            arrString += str(val) + ' '
        return arrString

    # Load training data from a file into array
    def loadTrainingData(self, path):
        training = []

        trainingFile = open(path, "r")

        for ram in trainingFile:
            data = [int(numeric_string) for numeric_string in ram.split(' ')[:-1]]
            training.append(torch.tensor(data, dtype=torch.uint8))
        
        return training

    # Load testing data from a file into array
    def loadTestingData(self, path):
        testing = []

        testingFile = open(path, "r")

        for num in testingFile.readline().split(' ')[:-1]:
            data = int(num)
            testing.append(torch.torch.LongTensor([data]))
        
        return testing

    def normalise(self, tensor):
        tensorSum = int(tensor.sum())
        return tensor / tensorSum

    # Based on where the ball landed in relation to the paddle, find the target
    def getTarget(self, paddleMid, ballMid):
        arr = []

        # If the ball hits the paddle, remain in the same place
        if abs(paddleMid - ballMid) < 7:
            arr = [1.0, 0.0, 0.0, 0.0]
        # paddle left of ball so move right
        if paddleMid < ballMid:
            arr = [0.0, 0.0, 1.0, 0.0]
        # paddle right of ball so move left
        elif paddleMid > ballMid:
            arr = [0.0, 0.0, 0.0, 1.0]
            
        return torch.tensor(arr, device=self.device)

    # Scale RAM values (tensor) to be between 0-1
    def scaleRAM(self, RAM):

        scaled_RAM = []

        for value in RAM:
            scaled_RAM.append(value.item() / 255)

        return torch.tensor(scaled_RAM, requires_grad=True).to(self.device)

    # Get the action the network takes
    def tensorAction(self, net, tensorIn):
        output = net(tensorIn)

        maxVal = torch.max(output, 0)
        return int(maxVal[1])

    # Take observation input and convert to scaled tensor
    def observationToTensor(self, observation):
        observation = self.scaleRAM(observation)
        return torch.tensor(observation, requires_grad=True, device=self.device)

    # Get the output from the network from a particular observation
    def getOutput(self, net, tensorIn):
        output = net(tensorIn)
        return output

    # Get the output from the network from a particular observation scaled
    def getScaledOutput(self, net, tensorIn):
        output = net(tensorIn)
        output = F.softmax(output)
        return output
    
    # Get the action the network takes based on an observation
    def observationAction(self, net, observation):
        observationTensor = torch.tensor(observation, device=self.device)

        paddleMid = int(observation[72]) + 10
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, paddleMid], device=self.device)
        outputs = net(tensorInput.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])

    def calculateTargetLoss(self, ballMid, paddleMid):
    
        distance = abs(ballMid - paddleMid)
        if distance < 7:
            loss = 0
        else: 
            loss = distance / 160
        return torch.tensor(loss)

    def applyLoss(self, outputs, targetLoss):
        loss = torch.mul(outputs, targetLoss)
        loss = torch.sum(torch.abs(loss))
        return loss
