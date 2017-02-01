import nengo
import numpy as np

from nengolib.signal import z

class State(object):
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.state = []
        self.action = []
    def get(self, index):
        if index < 0:
            index=0
        while index >= len(self.state):
            self.generate_next()
        return self.state[index], self.action[index]
    def generate_next(self):
        self.state.append('0')
        #if self.rng.rand() < 0.5:
        prob=0.7
        if len(self.state)%4==1:     # use this to alternate L, R
            self.action.append('L')
            self.action.append('L')
            if self.rng.rand()<prob:
                self.state.append('1')
            else:
                self.state.append('2')
        else:
            self.action.append('R')
            self.action.append('R')
            if self.rng.rand()<prob:
                self.state.append('2')
            else:
                self.state.append('1')
            
statemap = {
    '0': [1,0,0],
    '1': [0,1,0],
    '2': [0,0,1],
    }
    
actionmap = {
    'L': [1, 0],
    'R': [0, 1],
    }

state = State(seed=0)
t_isi = 0.1

model = nengo.Network()
with model:
    stim_state = nengo.Node(lambda t: statemap[state.get(int(t/t_isi))[0]]+actionmap[state.get(int(t/t_isi))[1]])
    #stim_state_delay = nengo.Node(lambda t: statemap[state.get(int(t/t_isi)-1)[0]])

    
    pre = nengo.Ensemble(n_neurons=200, dimensions=5, 
                        #intercepts=nengo.dists.Uniform(0,0.9),  # use this to sparsify pre
                        )
    
    post = nengo.Ensemble(n_neurons=200, dimensions=3)
    
    nengo.Connection(stim_state[:3], pre[:3], synapse=z**(-100))
    nengo.Connection(stim_state[3:], pre[3:], synapse=z**(-100))
    
    tau_slow=0.1
    c = nengo.Connection(pre, post, function=lambda x: [0,0,0],
                         learning_rule_type=nengo.PES(pre_tau=0.005, learning_rate=3e-5),
                         )

    error = nengo.Ensemble(n_neurons=200, dimensions=3)
    
    nengo.Connection(post, error, synapse=0.005)
    nengo.Connection(stim_state[:3], error, transform=-1)
    nengo.Connection(error, c.learning_rule)
    