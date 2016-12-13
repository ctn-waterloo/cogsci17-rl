import nengo
from nengo import spa
import numpy as np

DIM = 64

states = ['S0', 'S1', 'S2']

actions = ['L', 'R']

input_keys = ['S0*L', 'S0*R', 'S1*L', 'S1*R', 'S2*L', 'S2*R']
output_keys = ['0.7*S1 + 0.3*S2', '0.3*S1 + 0.7*S2', 'S0', 'S0', 'S0', 'S0']

vocab = spa.Vocabulary(dimensions=DIM)

# TODO: these vectors might need to be chosen in a smarter way
for sp in states+actions:
    vocab.parse(sp)

model = nengo.Network('RL', seed=13)
with model:

    model.assoc_mem = spa.AssociativeMemory(input_vocab=vocab,
                                            output_vocab=vocab,
                                            input_keys=input_keys,
                                            output_keys=output_keys,
                                            wta_output=True,
                                            threshold_output=True
                                           )

    model.state = spa.State(DIM, vocab=vocab)
    model.action = spa.State(DIM, vocab=vocab)
    model.probability = spa.State(DIM, vocab=vocab)

    # State and selected action convolved together
    #model.state_and_action = spa.State(DIM, vocab=vocab)

    model.cconv = nengo.networks.CircularConvolution(300, DIM)

    nengo.Connection(model.state.output, model.cconv.A)
    nengo.Connection(model.action.output, model.cconv.B)
    nengo.Connection(model.cconv.output, model.assoc_mem.input)
    nengo.Connection(model.assoc_mem.output, model.probability.input)

    # Semantic pointer for the Q values of each state
    # In the form of q0*S0 + q1*S1 + q2*S2
    model.q = spa.State(DIM, vocab=vocab)
    
    # Scalar reward value from the dot product of P and Q
    model.reward = nengo.Ensemble(100, 1)

    #TODO: figure out what the result of P.Q is used for
    model.prod = nengo.networks.Product(n_neurons=15*DIM, dimensions=DIM)

    nengo.Connection(model.probability.output, model.prod.A)
    nengo.Connection(model.q.output, model.prod.B)

    nengo.Connection(model.prod.output, model.reward,
                     transform=np.ones((1,DIM)))
