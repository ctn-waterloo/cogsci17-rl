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

# The ideal function that should be learned
def correct_mapping(x):
    state = x[:DIM]
    action = x[DIM:]

    closest_state = find_closest_vector(state, index_to_state_vector)
    closest_action = find_closest_vector(action, index_to_action_vector)

    if closest_state == 0:
        if closest_action == 0: # Left
            return index_to_state_vector[1]*.7 + index_to_state_vector[2]*.3
        elif closest_action == 1: # Right
            return index_to_state_vector[1]*.3 + index_to_state_vector[2]*.7
    else:
        # Always return to state 0 at this point
        return index_to_state_vector[0]

def initial_mapping(x):
    state = x[:DIM]
    action = x[DIM:]

    closest_state = find_closest_vector(state, index_to_state_vector)
    closest_action = find_closest_vector(action, index_to_action_vector)

    if closest_state == 0:
        if closest_action == 0: # Left
            return index_to_state_vector[1]*.5 + index_to_state_vector[2]*.5
        elif closest_action == 1: # Right
            return index_to_state_vector[1]*.5 + index_to_state_vector[2]*.5
    else:
        # Always return to state 0 at this point
        return index_to_state_vector[0]

#FIXME: this is currently hardcoded for only 5 dimensions
def make_probability(t, x):
    s0 = min(max(0, x[0]),1)
    s1 = min(max(0, x[1]),1)
    s2 = min(max(0, x[2]),1)
    #total = np.sum(x[0], x[1], x[2])
    total = s0 + s1 + s2
    
    if total > 0:
        return (s0/total, s1/total, s2/total, x[3], x[4])
    else:
        return x



def get_model(q_scaling=1, direct=False, p_learning=True, initialized=False,
              learning_rate=1e-4, forced_prob=False, intercept_dist=0):


    model = nengo.Network('RL P-learning', seed=13)
    
    if intercept_dist == 0:
        intercepts = nengo.dists.Uniform(-1,1)
    elif intercept_dist == 1:
        intercepts = AreaIntercepts(dimensions=DIM*2)
    elif intercept_dist == 1:
        intercepts = nengo.dists.Uniform(-.3,1)

    with model:
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        if direct:
            cfg[nengo.Ensemble].neuron_type = nengo.Direct()
            cfg[nengo.Connection].synapse = None

        # Model of the external environment
        agent = AgentSplit(vocab=vocab, time_interval=time_interval,
                      q_scaling=q_scaling)
        model.env = nengo.Node(agent,
                               size_in=1, size_out=DIM*4)

        with cfg:
            model.state = spa.State(DIM, vocab=vocab)
            model.action = spa.State(DIM, vocab=vocab)
            model.probability = spa.State(DIM, vocab=vocab)

            # The action that is currently being used along with the state to calculate value
            # If this matches with the actual action being taken, learning will happen (on the next step after a delay)
            model.calculating_action = spa.State(DIM, vocab=vocab)
            nengo.Connection(model.env[DIM*3:DIM*4], model.calculating_action.input)



        if p_learning:
            # State and selected action in one ensemble
            model.state_and_action = nengo.Ensemble(n_neurons=n_sa_neurons, dimensions=DIM*2, intercepts=intercepts)
            if initialized:
                function = correct_mapping
            else:
                function= initial_mapping
            conn = nengo.Connection(model.state_and_action, model.probability.input,
                                    function=function,
                                    learning_rule_type=nengo.PES(pre_synapse=z**(-int(time_interval*2*1000)),
                                                                 learning_rate=learning_rate),
                                   )
        else:
            with cfg:
                # State and selected action in one ensemble
                model.state_and_action = nengo.Ensemble(n_neurons=n_sa_neurons, dimensions=DIM*2, intercepts=intercepts)
                nengo.Connection(model.state_and_action, model.probability.input, function=correct_mapping)

        with cfg:
            nengo.Connection(model.state.output, model.state_and_action[:DIM])
            nengo.Connection(model.env[DIM*3:DIM*4], model.state_and_action[DIM:])
            
            # Scalar value from the dot product of P and Q
            model.value = nengo.Ensemble(100, 1, neuron_type=nengo.Direct())

            # Semantic pointer for the Q values of each state
            # In the form of q0*S0 + q1*S1 + q2*S2
            model.q = spa.State(DIM, vocab=vocab)

            model.prod = nengo.networks.Product(n_neurons=n_prod_neurons, dimensions=DIM)

            if forced_prob:
                normalized_prob = nengo.Node(make_probability, size_in=DIM, size_out=DIM)
                nengo.Connection(model.probability.output, normalized_prob, synapse=None)
                nengo.Connection(normalized_prob, model.prod.A, synapse=None)

            else:
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
            if p_learning:
                nengo.Connection(model.error_node, conn.learning_rule)

            #TODO: figure out which way the sign goes, one should be negative, and the other positive
            #TODO: figure out how to delay by one "time-step" correctly
            nengo.Connection(model.state.output, model.error.input, transform=-1)
            nengo.Connection(model.probability.output, model.error.input, transform=1,
                             synapse=z**(-int(time_interval*2*1000)))
                             #synapse=nengolib.synapses.PureDelay(500)) #500ms delay

            # Testing the delay synapse to make sure it works as expected
            model.state_delay_test = spa.State(DIM, vocab=vocab)
            nengo.Connection(model.state.output, model.state_delay_test.input,
                             synapse=z**(-int(time_interval*2*1000)))

            if direct:
                nengo.Connection(model.value, model.env, synapse=0.025)
            else:
                nengo.Connection(model.value, model.env, synapse=0.025)
    return model, agent
