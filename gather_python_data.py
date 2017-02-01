# Runs the pure python model a bunch of times and prints to a file that can be plotted with pandas

from __future__ import print_function  # Only needed for Python 2
from daw_mbf_1 import Agent
from calcStayProb import CalcStayProb
import sys
import time

num_runs = 40
num_steps = 40000

# Read option parameters from the command line
if len(sys.argv) == 3:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
if len(sys.argv) == 2:
    num_runs = int(sys.argv[1])

# {'alpha': 0.11039931648303387, 'noise': 0.045787897002043665}
#for alpha in [0.11039931648303387]:#[0.07785532763827441]:#[0.06803004214676607]:#[0.45553532900447363]:#[0.3566806]:#[0.3]:#[0.01,0.05, 0.1, .2, .3]:
#    for noise in [0.045787897002043665]:#[0.04313952374995614]:#[0.039566660494930024]:#[0.05866279768044559]:#[0.056433]:#[0.5]:#[0.05, 0.5]:
for alpha in [0.01,0.05, 0.1, .2, .3]:
    for noise in [0.05, 0.5]:
        outfile_name = 'data/out_py_r{0}_s{1}_a{2}_n{3}.txt'.format(num_runs, num_steps, alpha, noise)
        with open(outfile_name, 'w+') as outfile:
            for i in range(num_runs):
                print('{0}/{1} Runs'.format(i,num_runs))

                agent = Agent(alpha=alpha, noise=noise)
                temp_str = []

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
                calculator.doItAllString(temp_str, outfile)
        print(outfile_name)
