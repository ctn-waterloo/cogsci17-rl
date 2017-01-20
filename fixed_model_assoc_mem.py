# Transition probabilities are set and not learned
# Calculates the value for both actions from a given state

import nengo
from nengo import spa
import numpy as np
from environment import Environment
from modelbasednode import Agent
#import nengolib
from nengolib.signal import z
import scipy

DIM = 5#64

# Time between state transitions
time_interval = 0.1#0.5

states = ['S0', 'S1', 'S2']

actions = ['L', 'R']

input_keys = ['S0*L', 'S0*R', 'S1*L', 'S1*R', 'S2*L', 'S2*R']
output_keys = ['0.7*S1 + 0.3*S2', '0.3*S1 + 0.7*S2', 'S0', 'S0', 'S0', 'S0']

input_keys_left = ['S0', 'S1', 'S2']
output_keys_left = ['0.7*S1 + 0.3*S2', 'S0', 'S0']

input_keys_right = ['S0', 'S1', 'S2']
output_keys_right = ['0.3*S1 + 0.7*S2', 'S0', 'S0']

n_sa_neurons = DIM*2*15 # number of neurons in the state+action population
n_prod_neurons = DIM*15 # number of neurons in the product network

# Set all vectors to be orthogonal for now (easy debugging)
vocab = spa.Vocabulary(dimensions=DIM, randomize=False)

# TODO: these vectors might need to be chosen in a smarter way
for sp in states+actions:
    vocab.parse(sp)

class AreaIntercepts(nengo.dists.Distribution):
    dimensions = nengo.params.NumberParam('dimensions')
    base = nengo.dists.DistributionParam('base')

    def __init__(self, dimensions, base=nengo.dists.Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def __repr(self):
        return "AreaIntercepts(dimensions=%r, base=%r)" % (self.dimensions, self.base)

    def transform(self, x):
        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(1-scipy.special.betaincinv((self.dimensions+1)/2.0, 0.5, x+1))

    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for i in range(len(s)):
            s[i] = self.transform(s[i])
        return s

def ideal_error(t, x):
    state = x[:DIM]
    action = x[DIM:]

    res = np.zeros(DIM*2)

    action_index = find_closest_vector(action, index_to_action_vector)

    res[DIM*action_index:DIM*(action_index+1)] = state*-1

    return res


# takes a state index and returns the corresponding vector for the semantic pointer
index_to_state_vector = np.zeros((len(states), DIM))

# takes an action index and returns the corresponding vector for the semantic pointer
index_to_action_vector = np.zeros((len(actions), DIM))

# Fill in mapping data structures based on the vocab given
for i, vk in enumerate(vocab.keys):
    if vk in actions:
        index_to_action_vector[actions.index(vk)] = vocab.vectors[i]

    if vk in states:
        index_to_state_vector[states.index(vk)] = vocab.vectors[i]

def find_closest_vector(vec, index_to_vector):
    # Find the dot product with all other vectors
    distance = 0
    best_index = 0
    for i, v in enumerate(index_to_vector):
        d = np.dot(vec, v)
        if d > distance:
            distance = d
            best_index = i

    return best_index


neuron_type=nengo.Direct()
model = nengo.Network('RL P-learning', seed=13)
#model.config[nengo.Ensemble].neuron_type = nengo.Direct()
with model:
    cfg = nengo.Config(nengo.Ensemble)
    cfg[nengo.Ensemble].neuron_type = nengo.Direct()
    # Model of the external environment
    # Input: action semantic pointer
    # Output: current state semantic pointer
    #model.env = nengo.Node(Environment(vocab=vocab, time_interval=time_interval), size_in=DIM, size_out=DIM)
    model.env = nengo.Node(Agent(vocab=vocab, time_interval=time_interval),
                           size_in=2, size_out=DIM*3)

    model.state = spa.State(DIM, vocab=vocab)
    model.action = spa.State(DIM, vocab=vocab)
    #model.probability = spa.State(DIM, vocab=vocab)
    with cfg:
        model.probability_left = spa.State(DIM, vocab=vocab)
        model.probability_right = spa.State(DIM, vocab=vocab)
    model.state_ensemble = nengo.Ensemble(n_neurons=DIM*50,dimensions=DIM)
    
    model.assoc_mem_left = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=input_keys_left,
                                            output_keys=output_keys_left,
                                            wta_output=True,
                                            threshold_output=True
                                           )
    model.assoc_mem_right = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=input_keys_right,
                                            output_keys=output_keys_right,
                                            wta_output=True,
                                            threshold_output=True
                                           )
    
    nengo.Connection(model.env[DIM:DIM*2], model.assoc_mem_left.input)
    nengo.Connection(model.assoc_mem_left.output, model.probability_left.input)
    nengo.Connection(model.env[DIM:DIM*2], model.assoc_mem_right.input)
    nengo.Connection(model.assoc_mem_right.output, model.probability_right.input)

    # Semantic pointer for the Q values of each state
    # In the form of q0*S0 + q1*S1 + q2*S2
    model.q = spa.State(DIM, vocab=vocab)
    
    # Scalar reward value from the dot product of P and Q
    model.value = nengo.Ensemble(200, 2, neuron_type=neuron_type)
    

    with cfg:
        model.prod_left = nengo.networks.Product(n_neurons=50*DIM, dimensions=DIM)
        model.prod_right = nengo.networks.Product(n_neurons=50*DIM, dimensions=DIM)

    nengo.Connection(model.probability_left.output, model.prod_left.A)
    nengo.Connection(model.env[DIM*2:], model.prod_left.B)
    
    nengo.Connection(model.probability_right.output, model.prod_right.A)
    nengo.Connection(model.env[DIM*2:], model.prod_right.B)

    nengo.Connection(model.prod_left.output, model.value[0],
                     transform=np.ones((1,DIM)))
    nengo.Connection(model.prod_right.output, model.value[1],
                     transform=np.ones((1,DIM)))
    

    nengo.Connection(model.value, model.env)
    nengo.Connection(model.env[:DIM], model.action.input) # Purely for plotting
    nengo.Connection(model.env[DIM*2:], model.q.input) # Purely for plotting
    nengo.Connection(model.env[DIM:DIM*2], model.state.input)
