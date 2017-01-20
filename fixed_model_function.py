# Everything stuck in a function that can be run to get data
# Returns a model that will write to a particular file
# Learning the transition probabilities with the PES rule

import nengo
from nengo import spa
import numpy as np
from environment import Environment
from modelbasednode import Agent, AgentSplit
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

def selected_error(t, x):
    error = x[:DIM]
    action1 = x[DIM:DIM*2]
    action2 = x[DIM*2:]

    res = np.zeros(DIM)

    action_index1 = find_closest_vector(action1, index_to_action_vector)
    action_index2 = find_closest_vector(action2, index_to_action_vector)

    if action_index1 == action_index2:
        return error
    else:
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



def get_model(q_scaling=1, direct=False, p_learning=True):


    model = nengo.Network('RL P-learning', seed=13)
    #if direct:
    #    model.config[nengo.Ensemble].neuron_type = nengo.Direct()
    if direct:
        neuron_type = nengo.Direct()
    else:
        neuron_type = nengo.LIF()

    if p_learning:
        """
        raise NotImplementedError
        with model:

            # Model of the external environment
            # Input: action semantic pointer
            # Output: current state semantic pointer
            #model.env = nengo.Node(Environment(vocab=vocab, time_interval=time_interval), size_in=DIM, size_out=DIM)
            agent = Agent(vocab=vocab, time_interval=time_interval,
                          q_scaling=q_scaling)
            model.env = nengo.Node(agent,
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
            model.reward = nengo.Ensemble(100, 1, neuron_type=neuron_type)

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

        return model, agent
        """
        with model:

            # Model of the external environment
            agent = AgentSplit(vocab=vocab, time_interval=time_interval,
                          q_scaling=q_scaling)
            model.env = nengo.Node(agent,
                                   size_in=1, size_out=DIM*4)

            model.state = spa.State(DIM, vocab=vocab)
            model.action = spa.State(DIM, vocab=vocab)
            model.probability = spa.State(DIM, vocab=vocab)

            # The action that is currently being used along with the state to calculate value
            # If this matches with the actual action being taken, learning will happen (on the next step after a delay)
            model.calculating_action = spa.State(DIM, vocab=vocab)
            nengo.Connection(model.env[DIM*3:DIM*4], model.calculating_action.input)


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
            
            # Scalar value from the dot product of P and Q
            model.value = nengo.Ensemble(100, 1)

            #TODO: figure out what the result of P.Q is used for
            model.prod = nengo.networks.Product(n_neurons=n_prod_neurons, dimensions=DIM)

            nengo.Connection(model.probability.output, model.prod.A)
            #nengo.Connection(model.q.output, model.prod.B)
            nengo.Connection(model.env[DIM*2:DIM*3], model.prod.B)

            nengo.Connection(model.prod.output, model.value,
                             transform=np.ones((1,DIM)))

            #TODO: doublecheck that this is the correct way to connect things
            nengo.Connection(model.env[DIM:DIM*2], model.state.input)

            #TODO: need to set up error signal and handle timing
            model.error = spa.State(DIM, vocab=vocab)
            ##nengo.Connection(model.error.output, conn.learning_rule)
            
            model.error_node = nengo.Node(selected_error,size_in=DIM*3, size_out=DIM)
            nengo.Connection(model.error.output, model.error_node[:DIM])
            nengo.Connection(model.action.output, model.error_node[DIM:DIM*2])
            nengo.Connection(model.calculating_action.output, model.error_node[DIM*2:DIM*3])
            nengo.Connection(model.error_node, conn.learning_rule)

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

            nengo.Connection(model.value, model.env)
            nengo.Connection(model.env[:DIM], model.action.input) # Purely for plotting
            nengo.Connection(model.env[DIM*2:DIM*3], model.q.input) # Purely for plotting
        return model, agent
    else:
        if direct:
            neuron_type = nengo.Direct()
        else:
            neuron_type = nengo.LIF()
        with model:
            agent = Agent(vocab=vocab, time_interval=time_interval,
                          q_scaling=q_scaling)
            model.env = nengo.Node(agent,
                                   size_in=2, size_out=DIM*3)

            model.state = spa.State(DIM, vocab=vocab)
            model.action = spa.State(DIM, vocab=vocab)
            cfg = nengo.Config(nengo.Ensemble)
            cfg[nengo.Ensemble].neuron_type = neuron_type
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
            #model.value = nengo.Ensemble(200, 2, neuron_type=neuron_type)
            model.value = nengo.Ensemble(200, 2, neuron_type=nengo.Direct())

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
        return model, agent
