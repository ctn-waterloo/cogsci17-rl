# Runs the nengo model a bunch of times and prints to a file that can be plotted with pandas

from __future__ import print_function  # Only needed for Python 2
#from full_model_function import get_model
from fixed_model_function import get_model
from calcStayProb import CalcStayProb
import sys
import time
import nengo

num_runs = 3#100
num_steps = 40#40000
tf_name = 'temp_file.txt'

# Running some ensembles in Direct mode
direct = False#True

# If the transition probabilities should be learned
p_learning = False

# Read option parameters from the command line
if len(sys.argv) == 5:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
    direct = sys.argv[3] == 'True'
    p_learning = sys.argv[4] == 'True'
if len(sys.argv) == 4:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
    direct = sys.argv[3] == 'True'
if len(sys.argv) == 3:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
if len(sys.argv) == 2:
    num_runs = int(sys.argv[1])

# Day-Hour:Minute
date_time_string = time.strftime("%d-%H:%M")

outfile_name = 'data/out_nengo_r{0}_s{1}_d{2}_p{3}_{4}.txt'.format(num_runs, num_steps, direct, p_learning, date_time_string)
data_file_name = 'data/tmp_data_nengo_r{0}_s{1}_d{2}_p{3}_{4}.txt'.format(num_runs, num_steps, direct, p_learning, date_time_string)
with open(outfile_name, 'w+') as outfile:

    for i in range(num_runs):
        print('{0}/{1} Runs'.format(i+1,num_runs))

        model, agent = get_model(direct=direct, p_learning=p_learning)

        sim = nengo.Simulator(model)
        if p_learning: # The current version of p_learning needs to run through twice for each step
            sim.run(num_steps*2*.1)
        else:
            sim.run(num_steps*.1)
        temp_str_list = agent.result_string


        calculator = CalcStayProb()
        calculator.doItAllString(temp_str_list, outfile)

with open(data_file_name, 'w+') as data_file:
    data_file.write('\n'.join(temp_str_list))
print(outfile_name)
