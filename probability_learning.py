# Learning the transition probabilities with the PES rule

import nengo
from nengo import spa
import numpy as np
from environment import Environment

DIM = 64

states = ['S0', 'S1', 'S2']

actions = ['L', 'R']

n_sa_neurons = DIM*2*15 # number of neurons in the state+action population
n_prod_neurons = DIM*15 # number of neurons in the product network

vocab = spa.Vocabulary(dimensions=DIM)

# TODO: these vectors might need to be chosen in a smarter way
for sp in states+actions:
    vocab.parse(sp)

model = nengo.Network('RL P-learning', seed=13)
with model:

    # Model of the external environment
    # Input: action semantic pointer
    # Output: current state semantic pointer
    model.env = nengo.Node(Environment(vocab=vocab), size_in=DIM, size_out=DIM)

    model.state = spa.State(DIM, vocab=vocab)
    model.action = spa.State(DIM, vocab=vocab)
    model.probability = spa.State(DIM, vocab=vocab)

    # The state that is actually entered after an action is performed
    model.resulting_state = spa.State(DIM, vocab=vocab)

    # State and selected action in one ensemble
    model.state_and_action = nengo.Ensemble(n_neurons=n_sa_neurons, dimensions=DIM*2)

    #model.cconv = nengo.networks.CircularConvolution(300, DIM)

    nengo.Connection(model.state.output, model.state_and_action[:DIM])
    nengo.Connection(model.action.output, model.state_and_action[DIM:])
    nengo.Connection(model.state_and_action, model.probability.input, function=lambda x: [0]*DIM)


    # Semantic pointer for the Q values of each state
    # In the form of q0*S0 + q1*S1 + q2*S2
    model.q = spa.State(DIM, vocab=vocab)
    
    # Scalar reward value from the dot product of P and Q
    model.reward = nengo.Ensemble(100, 1)

    #TODO: figure out what the result of P.Q is used for
    model.prod = nengo.networks.Product(n_neurons=n_prod_neurons, dimensions=DIM)

    nengo.Connection(model.probability.output, model.prod.A)
    nengo.Connection(model.q.output, model.prod.B)

    nengo.Connection(model.prod.output, model.reward,
                     transform=np.ones((1,DIM)))

    #TODO: doublecheck that this is the correct way to connect things
    nengo.Connection(model.env, model.state.input)
    nengo.Connection(model.action.output, model.env)

    #TODO: need to set up error signal and handle timing