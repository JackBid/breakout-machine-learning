from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent

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
    if sys.argv[3].lower() == 'true':
        render = True

if agentName.lower() == 'supervised':
    supervisedAgent = SupervisedAgent()
    supervisedAgent.train(trainingLength)
elif agentName.lower() == 'evolvedreinforced':
    reinforcementAgent = EvolvedReinforcedAgent(debug=False, render=render)
    reinforcementAgent.train(trainingLength)
else:
    print('unknown agent.')
