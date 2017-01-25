# Learning the transition probabilities with the PES rule
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

n_sa_neurons = DIM*2*15 # number of neurons in the state+action population
n_prod_neurons = DIM*50 # number of neurons in the product network

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
    model.probability_left = spa.State(DIM, vocab=vocab)
    model.probability_right = spa.State(DIM, vocab=vocab)
    model.combined_probability = nengo.Ensemble(n_neurons=DIM*2*50,
                                                dimensions=DIM*2,
                                               )#neuron_type=neuron_type)
    # Initialize transition probability estimates to 50% each
    #init_node = nengo.Node(lambda t: 0 if t < 2.5 else 0)
    #nengo.Connection(init_node, model.probability_left.input[:3], transform=[[1],[1],[1]]) # super hacky for now
    #nengo.Connection(init_node, model.probability_right.input[:3], transform=[[1],[1],[1]]) # super hacky for now
    
    model.state_ensemble = nengo.Ensemble(n_neurons=DIM*50,dimensions=DIM)
    

    nengo.Connection(model.combined_probability[:DIM], model.probability_left.input)
    nengo.Connection(model.combined_probability[DIM:], model.probability_right.input)

    # State and selected action in one ensemble
    model.state_and_action = nengo.Ensemble(n_neurons=n_sa_neurons, dimensions=DIM*2, intercepts=AreaIntercepts(dimensions=DIM*2))
    nengo.Connection(model.state.output, model.state_ensemble)
    #nengo.Connection(model.state.output, model.state_and_action[:DIM])
    #nengo.Connection(model.env[:DIM], model.state_and_action[DIM:])
    conn = nengo.Connection(model.state_ensemble, model.combined_probability,
                            function=lambda x: [0]*DIM*2,
                            learning_rule_type=nengo.PES(pre_synapse=z**(-int(time_interval*1000))),
                           )


    # Semantic pointer for the Q values of each state
    # In the form of q0*S0 + q1*S1 + q2*S2
    model.q = spa.State(DIM, vocab=vocab)
    
    # Scalar reward value from the dot product of P and Q
    model.value = nengo.Ensemble(200, 2, neuron_type=neuron_type)

    #TODO: figure out what the result of P.Q is used for
    model.prod_left = nengo.networks.Product(n_neurons=n_prod_neurons, dimensions=DIM)
    model.prod_right = nengo.networks.Product(n_neurons=n_prod_neurons, dimensions=DIM)

    nengo.Connection(model.probability_left.output, model.prod_left.A)
    nengo.Connection(model.probability_right.output, model.prod_right.A)
    
    nengo.Connection(model.env[DIM*2:], model.prod_left.B)
    nengo.Connection(model.env[DIM*2:], model.prod_right.B)

    nengo.Connection(model.prod_left.output, model.value[0],
                     transform=np.ones((1,DIM)))
    nengo.Connection(model.prod_right.output, model.value[1],
                     transform=np.ones((1,DIM)))

    nengo.Connection(model.env[DIM:DIM*2], model.state.input)

    #TODO: need to set up error signal and handle timing
    #model.error = spa.State(DIM, vocab=vocab)
    model.error = nengo.Ensemble(n_neurons=DIM*2*50, dimensions=DIM*2)
    
    #nengo.Connection(model.error.output, conn.learning_rule)
    model.error_node = nengo.Node(ideal_error,size_in=DIM*2, size_out=DIM*2)
    nengo.Connection(model.error_node, conn.learning_rule)
    nengo.Connection(model.state.output, model.error_node[:DIM],
                     synapse=z**(-int(time_interval*1000)))
    nengo.Connection(model.action.output, model.error_node[DIM:],
                     synapse=z**(-int(time_interval*1000)))


    #TODO: figure out which way the sign goes, one should be negative, and the other positive
    #TODO: figure out how to delay by one "time-step" correctly
    ##nengo.Connection(model.state.output, model.error.input, transform=-1)
    nengo.Connection(model.error_node, model.error)#, transform=-1)
    nengo.Connection(model.combined_probability, model.error, transform=1,
                     synapse=z**(-int(time_interval*1000)))
                     #synapse=nengolib.synapses.PureDelay(500)) #500ms delay

    # Testing the delay synapse to make sure it works as expected
    model.state_delay_test = spa.State(DIM, vocab=vocab)
    nengo.Connection(model.state.output, model.state_delay_test.input,
                     synapse=z**(-int(time_interval*1000)))

    nengo.Connection(model.value, model.env)
    nengo.Connection(model.env[:DIM], model.action.input) # Purely for plotting
    nengo.Connection(model.env[DIM*2:], model.q.input) # Purely for plotting
