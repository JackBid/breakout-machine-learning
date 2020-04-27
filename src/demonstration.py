from agents.trivial import TrivialAgent
from agents.supervised import SupervisedAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent
from simulation import Simulation
import os
import sys

def agentsComparison():
    supervisedSim = Simulation('supervised', render = True, record = False)
    reinforcedSim = Simulation('evolvedreinforced', render = True, record = False)
    supervisedSim.run(5)
    reinforcedSim.run(5)


agentsComparison()