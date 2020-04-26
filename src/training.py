from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
from agents.reinforced import BasicReinforcedAgent
from agents.transfer import TransferAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent
from agents.dqnAgent import DQNAgent
from agents.util import Util
import numpy as np
import torch
import config
import gym
import time
import os
import sys

# Global variable for command line arguments
render = False
agentName = 'evolvedReinforced'
trainingLength = 100

# Just agent name provided
if len(sys.argv) == 2:
    agentName = sys.argv[1]
# Agent name and simulation length provided
elif len(sys.argv) == 3:
    agentName = sys.argv[1]
    trainingLength = int(sys.argv[2])
# All three arguments provided
else:
    agentName = sys.argv[1]
    trainingLength = int(sys.argv[2])
    render = bool(sys.argv[3])

if agentName.lower() == 'supervised':
    supervisedAgent = SupervisedAgent()
    supervisedAgent.train(trainingLength)
elif agentName.lower() == 'evolvedreinforced':
    reinforcementAgent = EvolvedReinforcedAgent(render)
    reinforcementAgent.train(trainingLength)
else:
    print('unknown agent.')
