import sys
import os
sys.path.append("..")

from agents.supervised import SupervisedAgent
from agents.evolvedReinforced import EvolvedReinforcedAgent
from simulation import Simulation

os.chdir('..')

def supervisedLiveProgress(iterations):
    sim = Simulation('supervised', render=True, debug=False)
    scores = []

    for i in range(iterations):
        scores.append(sim.run(1)[0])
        print(scores)

supervisedLiveProgress(5)
