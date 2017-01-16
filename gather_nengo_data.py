# Runs the nengo model a bunch of times and prints to a file that can be plotted with pandas

from __future__ import print_function  # Only needed for Python 2
from full_model_function import get_model
from calcStayProb import CalcStayProb
import sys
import time
import nengo

num_runs = 3#100
num_steps = 40#40000
tf_name = 'temp_file.txt'

# Read option parameters from the command line
if len(sys.argv) == 3:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
if len(sys.argv) == 2:
    num_runs = int(sys.argv[1])

outfile_name = 'data/out_nengo_r{0}_s{1}.txt'.format(num_runs, num_steps)
with open(outfile_name, 'w+') as outfile:

    for i in range(num_runs):
        print('{0}/{1} Runs'.format(i,num_runs))

        model, agent = get_model()

        sim = nengo.Simulator(model)
        sim.run(num_steps*.1)
        temp_str = agent.result_string


        calculator = CalcStayProb()
        calculator.doItAllString(temp_str, outfile)
