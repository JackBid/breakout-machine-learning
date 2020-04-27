import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets import FullyConnected, Test
from agents.util import Util

import os

class SupervisedAgent():
    """ Supervised Learning agent

    Trained using training and testing data generated from expert system
    """

    def __init__(self, load=True, save=True, saveFile=''):

        self.load = load
        self.save = save

        # object for utiliy functions
        self.util = Util()

        # load correct network weights
        if saveFile == '':
            self.saveFile = '../res/models/supervised.pth'
        else:
            self.saveFile = '../res/models/' + saveFile + '.pth'
        
        # Create network and load weights
        self.net = FullyConnected(128, 10)
        self.net = self.net.float()
        if self.load:
            self.net.load_state_dict(torch.load(self.saveFile))

        # Load training and testing data
        self.trainingData = self.util.loadTrainingData('../res/training data/evolvedObservation.txt')
        self.testingData = self.util.loadTestingData('../res/training data/evolvedAction.txt')

        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())


    def train(self, epochs):
        """ Train the agent for a given number of epochs

        epochs : int = number of training epochs

        """

        for epoch in range(0, epochs):

            running_loss = 0.0

            for i in range(0, len(self.trainingData)):

                self.optimizer.zero_grad()

                # Get network output and target
                tensorInput = self.trainingData[i]
                outputs = self.net(tensorInput.float())
                outputs = outputs.unsqueeze(dim=0)
                target = self.testingData[i].clone()

                # Calcuate loss between network output and target from testing dataset
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                # Calculate running loss and print every 2000 mini-batches
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                    if self.save:
                        torch.save(self.net.state_dict(), self.saveFile)
    
    def observationAction(self, observation):
        """ Get the agents action 

        observation : list = observation representing game state

        """

        observationTensor = torch.tensor(observation)

        outputs = self.net(observationTensor.float())

        maxVal = torch.max(outputs, 0)
        return int(maxVal[1])

