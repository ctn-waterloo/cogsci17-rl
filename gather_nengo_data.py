# Runs the nengo model a bunch of times and prints to a file that can be plotted with pandas

from __future__ import print_function  # Only needed for Python 2
#from full_model_function import get_model
#from fixed_model_function import get_model
from complete_model_function import get_model
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

# If the transition probabilities of learning start at the correct value rather than 0
initialized = False

forced_prob = False

default_intercepts = True

learning_rate = 1e-4

# Read option parameters from the command line
if len(sys.argv) == 9:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
    direct = sys.argv[3] == 'True'
    p_learning = sys.argv[4] == 'True'
    initialized = sys.argv[5] == 'True'
    forced_prob = sys.argv[6] == 'True'
    default_intercepts = sys.argv[7] == 'True'
    learning_rate = float(sys.argv[8])
else:
    print("Not all arguments specified")
if len(sys.argv) == 7:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
    direct = sys.argv[3] == 'True'
    p_learning = sys.argv[4] == 'True'
    initialized = sys.argv[5] == 'True'
    learning_rate = float(sys.argv[6])
if len(sys.argv) == 6:
    num_runs = int(sys.argv[1])
    num_steps = int(sys.argv[2])
    direct = sys.argv[3] == 'True'
    p_learning = sys.argv[4] == 'True'
    initialized = sys.argv[5] == 'True'
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

def l(b):
    if b:
        return 'T'
    else:
        return 'F'

# Day-Hour:Minute
date_time_string = time.strftime("%d-%H:%M")

suffix = 'nengo_r{0}_s{1}_d{2}_p{3}_i{4}_ps{5}_int{6}_l{7}_{8}.txt'.format(num_runs, num_steps, l(direct), l(p_learning), l(initialized), 
                                                                           l(forced_prob), l(default_intercepts), learning_rate, date_time_string)

outfile_name = 'data/out_' + suffix
data_file_name = 'data/tmp_data_' + suffix
with open(outfile_name, 'w+') as outfile:

    for i in range(num_runs):
        print('{0}/{1} Runs'.format(i+1,num_runs))

        model, agent = get_model(direct=direct, p_learning=p_learning, initialized=initialized, learning_rate=learning_rate,
                                 forced_prob=forced_prob, default_intercepts=default_intercepts)

        sim = nengo.Simulator(model)
        # The current version of p_learning needs to run through twice for each step
        sim.run(num_steps*2*.1)
        temp_str_list = agent.result_string


        calculator = CalcStayProb()
        calculator.doItAllString(temp_str_list, outfile)

with open(data_file_name, 'w+') as data_file:
    data_file.write('\n'.join(temp_str_list))
print(outfile_name)
