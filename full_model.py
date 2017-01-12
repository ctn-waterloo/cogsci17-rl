# Learning the transition probabilities with the PES rule

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

model = nengo.Network('RL P-learning', seed=13)
with model:

    # Model of the external environment
    # Input: action semantic pointer
    # Output: current state semantic pointer
    #model.env = nengo.Node(Environment(vocab=vocab, time_interval=time_interval), size_in=DIM, size_out=DIM)
    model.env = nengo.Node(Agent(vocab=vocab, time_interval=time_interval),
                           size_in=1, size_out=DIM*3)

    model.state = spa.State(DIM, vocab=vocab)
    model.action = spa.State(DIM, vocab=vocab)
    model.probability = spa.State(DIM, vocab=vocab)

    # State and selected action in one ensemble
    model.state_and_action = nengo.Ensemble(n_neurons=n_sa_neurons, dimensions=DIM*2, intercepts=AreaIntercepts(dimensions=DIM*2))

    #model.cconv = nengo.networks.CircularConvolution(300, DIM)

    nengo.Connection(model.state.output, model.state_and_action[:DIM])
    nengo.Connection(model.env[:DIM], model.state_and_action[DIM:])
    conn = nengo.Connection(model.state_and_action, model.probability.input,
                            function=lambda x: [0]*DIM,
                            learning_rule_type=nengo.PES(pre_synapse=z**(-int(time_interval*1000))),
                           )


    # Semantic pointer for the Q values of each state
    # In the form of q0*S0 + q1*S1 + q2*S2
    model.q = spa.State(DIM, vocab=vocab)
    
    # Scalar reward value from the dot product of P and Q
    model.reward = nengo.Ensemble(100, 1)

    #TODO: figure out what the result of P.Q is used for
    model.prod = nengo.networks.Product(n_neurons=n_prod_neurons, dimensions=DIM)

    nengo.Connection(model.probability.output, model.prod.A)
    #nengo.Connection(model.q.output, model.prod.B)
    nengo.Connection(model.env[DIM*2:], model.prod.B)

    nengo.Connection(model.prod.output, model.reward,
                     transform=np.ones((1,DIM)))

    #TODO: doublecheck that this is the correct way to connect things
    nengo.Connection(model.env[DIM:DIM*2], model.state.input)

    #TODO: need to set up error signal and handle timing
    model.error = spa.State(DIM, vocab=vocab)
    nengo.Connection(model.error.output, conn.learning_rule)

    #TODO: figure out which way the sign goes, one should be negative, and the other positive
    #TODO: figure out how to delay by one "time-step" correctly
    nengo.Connection(model.state.output, model.error.input, transform=-1)
    nengo.Connection(model.probability.output, model.error.input, transform=1,
                     synapse=z**(-int(time_interval*1000)))
                     #synapse=nengolib.synapses.PureDelay(500)) #500ms delay

    # Testing the delay synapse to make sure it works as expected
    model.state_delay_test = spa.State(DIM, vocab=vocab)
    nengo.Connection(model.state.output, model.state_delay_test.input,
                     synapse=z**(-int(time_interval*1000)))

    nengo.Connection(model.reward, model.env)
    nengo.Connection(model.env[:DIM], model.action.input) # Purely for plotting
    nengo.Connection(model.env[DIM*2:], model.q.input) # Purely for plotting
