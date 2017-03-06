# Runs the nengo model a bunch of times and prints to a file that can be plotted with pandas

from __future__ import print_function  # Only needed for Python 2
#from full_model_function import get_model
#from fixed_model_function import get_model
from complete_model_function import get_model
from calcStayProb import CalcStayProb
import sys
import time
import nengo
import os
import argparse

parser = argparse.ArgumentParser(description='Run the neural model-based reinforcement learning code and save the output')

parser.add_argument('--label', dest='label', type=str, default='', help='Optional label for further differentiating different runs')
parser.add_argument('--runs', dest='num_runs', type=int, default=10, help='Number of instances of the model to run')
parser.add_argument('--steps', dest='num_steps', type=int, default=20000, help='Number of steps to run each model for (default=20000)')
parser.add_argument('--direct', dest='direct', action='store_true', help='Run the model in direct mode or not (default=False)')
parser.add_argument('--p_learning', dest='p_learning', action='store_true', help='Set learning the state transition probabilities or not (if not set = False)')
parser.add_argument('--noinit', dest='initialized', action='store_false', help='If the transition probabilities are set to 50/50 or not (70/30). This flag should not be set if learning is not enabled, and should ideally be set if it is enabled (if not set=70/30)')
parser.add_argument('--synapse', dest='synapse', type=float, default=0.0, help='The synapse on the connection from value back to the environment (default=0.0)')
parser.add_argument('--dim', dest='dimensionality', type=int, default=5, help='Number of dimensions for the vocab (default=5)')
parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-4, help='The learning rate for the transition probabilities (default=1e-4)')
parser.add_argument('--intercepts', dest='intercept_dist', type=int, default=0, help='Type of intercepts to use for the population that is being learned from. 0 -> default, 1 -> AreaIntercepts, 2-> Uniform(-0.3, 1)')
#TODO: allow forcing probability to work for higher dimensions
parser.add_argument('--forcedprob', dest='forced_prob', action='store_true', help='If the output of the state transition calculation is forced to be a probability that sums to 1. NOTE: currently only supporting dim=5 (default=False)')
# original model used default nengo_seed of 13
parser.add_argument('--seed', dest='nengo_seed', type=int, default=1, help='Set nengo seed (default=1)')
parser.add_argument('--t_interval', dest='t_interval', type=float, default=0.1, help='Time between state transitions (default=0.1s)')
parser.add_argument('--valtoenv', dest='valtoenv', action='store_true', help='Use synapse of 0.025 for value to environment connection (if set)')


args = parser.parse_args()

num_runs = args.num_runs
num_steps = args.num_steps

label = args.label

# Set synapse of value to environment connection to 0.025
valtoenv = args.valtoenv

# Running some ensembles in Direct mode
direct = args.direct

# Nengo seed
nengo_seed = args.nengo_seed

# Time interval for presenting one state (time between state transitions)
t_interval = args.t_interval

# If the transition probabilities should be learned
p_learning = args.p_learning

# If the transition probabilities of learning start at the correct value rather than 0
initialized = args.initialized

forced_prob = args.forced_prob

intercept_dist = args.intercept_dist

# The synapse on the connection from value to env
synapse = args.synapse

learning_rate = args.learning_rate

dimensionality = args.dimensionality

def l(b):
    if b:
        return 'T'
    else:
        return 'F'

# Day-Hour:Minute
date_time_string = time.strftime("%b-%d-%H:%M")

suffix = 'nengo_r{0}_s{1}_d{2}_p{3}_i{4}_ps{5}_int{6}_sy{7}_dim{8}_l{9}_ns{10}_t{11}_v{12}_{13}_{14}'.format(num_runs, 
                                                                            num_steps, l(direct), l(p_learning), l(initialized), 
                                                                            l(forced_prob), intercept_dist, synapse, dimensionality, learning_rate, 
                                                                            nengo_seed, t_interval, valtoenv, date_time_string, label)

outfile_name = 'data/out_' + suffix + '.txt'
raw_data_dir = 'data/raw_data_' + suffix

if not os.path.exists('data'):
        os.makedirs('data')

if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

with open(outfile_name, 'w+') as outfile:

    for i in range(num_runs):
        print('{0}/{1} Runs'.format(i+1,num_runs))

        model, agent = get_model(direct=direct, p_learning=p_learning, initialized=initialized, learning_rate=learning_rate,
                                 forced_prob=forced_prob, intercept_dist=intercept_dist, synapse=synapse, dimensionality=dimensionality,
                                 nengo_seed=nengo_seed, t_interval = t_interval, valtoenv=valtoenv)

        sim = nengo.Simulator(model)
        # The current version of p_learning needs to run through twice for each step
        sim.run(num_steps*2*t_interval)
        temp_str_list = agent.result_string


        calculator = CalcStayProb()
        calculator.doItAllString(temp_str_list, outfile)

        # Saving data from each individual run to a separate file within a folder
        data_file_name = 'data/raw_data_' + suffix + '/' + str(i+1) + '.txt'
        with open(data_file_name, 'w+') as data_file:
            data_file.write('\n'.join(temp_str_list))
print(outfile_name)
