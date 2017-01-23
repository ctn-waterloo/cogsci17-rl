# Use hyperopt to figure out the parameters of the python code to get the
# correct magnitudes in the plots

from __future__ import print_function  # Only needed for Python 2
from daw_mbf_1 import Agent
from calcStayProb import CalcStayProb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import cPickle as pickle

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

# 'Ideal' values for the four bars
ideal = np.array([.774, .691, .697, .779])

# These parameters should be chosen and not optimized
num_runs = 40#100
num_steps = 40000#40000
max_evals = 1200

#def objective(alpha, noise):
def objective(args):
    alpha = args['alpha']
    noise = args['noise']
    results = np.zeros((num_runs,4))
    for i in range(num_runs):

        agent = Agent(alpha=alpha, noise=noise)
        temp_str = []

        #print "firstStageChoice secondStage secondStageChoice finalReward"
        firstStageChoice = None
        secondStage = None
        secondStageChoice = None
        finalReward = None
        for step in range(num_steps): # Repeat (for each step of episode):
            if agent.oneStep() == None:
                print ("oneStep broke")
                break
            if step%2 == 0: # in stage 1
                firstStageChoice = agent.getLastAction()
                secondStage = agent.getCurrBoardState()
            else: # in stage 2
                secondStageChoice = agent.getLastAction()
                finalReward = agent.getCurrReward()
                temp_str.append('{0} {1} {2} {3}'.format(firstStageChoice, secondStage, secondStageChoice, finalReward))


        calculator = CalcStayProb()
        results[i,:] = np.array(calculator.doItAllString(temp_str, return_value=True))

    avg = np.mean(results, axis=0)

    return {'loss': np.sqrt(np.mean((avg - ideal)**2)), 'status':STATUS_OK}

# Load from a previous run if possible
try:
    trials = pickle.load(open('hyperopt_data.p', 'rb'))['trials']
    previous_evals = len(trials)
except:
    trials = Trials()
    previous_evals = 0

#TODO: try different distributions
space = {'alpha':hp.uniform('alpha', 0, 1),
         'noise':hp.uniform('noise', 0, 1)}

for i in range(previous_evals, max_evals + previous_evals):
    print("Eval {0}/{1}".format(i+1, max_evals + previous_evals))
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=i+1)

print(best)
pickle.dump({'best':best, 'trials':trials}, open('hyperopt_data.p', 'wb'))

