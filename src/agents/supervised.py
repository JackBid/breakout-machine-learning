import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 6) 
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        #print(x)
        x = F.relu(self.fc1(x))
       # print(x)
        x = F.relu(self.fc2(x))
      #  print(x)
        return x

class SupervisedAgent():

    def __init__(self):
        self.net = Net()
        self.net = self.net.float()
        self.net.load_state_dict(torch.load('../res/models/supervised_net.pth'))

        self.trainingData = self.loadTrainingData('../res/training data/ram.txt')
        self.testingData = self.loadTestingData('../res/training data/action.txt')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())

    def loadTrainingData(self, path):
        training = []

        trainingFile = open(path, "r")

        for ram in trainingFile:
            data = [int(numeric_string) for numeric_string in ram.split(' ')[:-1]]
            training.append(torch.tensor(data, dtype=torch.uint8))
        
        return training

    def loadTestingData(self, path):
        testing = []

        testingFile = open(path, "r")

        for num in testingFile.readline().split(' ')[:-1]:
            data = int(num)
            testing.append(torch.torch.LongTensor([data]))
        
        return testing

    def train(self):

        for epoch in range(0, 3):
            running_loss = 0.0
            for i in range(0, len(self.trainingData)):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                paddleMid = int(self.trainingData[i][72]) + 13
                ballMid = int(self.trainingData[i][99]) + 1
                ballY = int(self.trainingData[i][101])
                
                tensorInput = torch.tensor([ballMid, ballY, paddleMid])
                outputs = self.net(tensorInput.float())

                target = self.testingData[i].clone()

                #outputs = F.softmax(outputs, dim=0)
                outputs = outputs.unsqueeze(dim=0)

                #print(outputs)
                #print(target)
                #print()

                loss = self.criterion(outputs, target)
                #print('loss: ' + str(loss))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                    PATH = '../res/models/supervised_net.pth'
                    torch.save(self.net.state_dict(), PATH)
    
    def action(self, observation):
        observationTensor = torch.tensor(observation)

        paddleMid = int(observation[72]) + 13
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        tensorInput = torch.tensor([ballMid, ballY, paddleMid])
        outputs = self.net(tensorInput.float())

        print(outputs)

        maxVal = torch.max(outputs, 0)
        print(maxVal[1])

        return int(maxVal[1])





