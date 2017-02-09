import nengo
import numpy as np
import nengo.spa as spa

D = 5
vocab = spa.Vocabulary(D, randomize=False)
vocab.parse('S0+SA+SB+L+R')
T_interval = 0.5
choice_noise = 0.5
alpha = 0.1
N_state_action = 500


class Environment(object):
    def __init__(self, vocab, seed):
        self.vocab = vocab
        self.state = 'S0'
        self.consider_action = 'L'
        self.q = np.zeros((3,2)) + np.inf  # we don't actually need Q(S0)!!
                                           # so maybe it could be removed?
        self.most_recent_action = 'L'
        self.values = np.zeros(2)
        self.value_wait_times = [T_interval/2, T_interval]
        self.n_intervals = 0
        self.rng = np.random.RandomState(seed=seed)
        self.reward_prob = self.rng.uniform(0.25, 0.75, size=(2,2))
        
    def node_function(self, t, value):
        if t >= self.value_wait_times[0]:
            self.values[0] = value
            self.value_wait_times[0] = (self.n_intervals+1.5)*T_interval
            self.consider_action = 'R'
        if t >= self.value_wait_times[1]:
            self.values[1] = value
            self.value_wait_times[1] = (self.n_intervals+2)*T_interval
            
            self.choose_action()
            self.n_intervals += 1
            self.consider_action = 'L'
            
        s = self.vocab.parse(self.state).v
        
        # replace infinities with 0
        q = np.max(np.where(self.q==np.inf, 0, self.q), axis=1)
        
        a = self.vocab.parse(self.consider_action).v
        return np.hstack([s, a, q])
        
        
    def choose_action(self):
        
        if self.state == 'S0':
            chosen = self.softmax(self.values)
            if chosen == 0:
                if self.rng.rand()<0.7:
                    self.state = 'SA'
                else:
                    self.state = 'SB'
            else:
                if self.rng.rand()<0.7:
                    self.state = 'SB'
                else:
                    self.state = 'SA'
        else:
            q_index = 1 if self.state=='SA' else 2
            chosen = self.softmax(self.q[q_index])
            p = self.reward_prob[0 if self.state=='SA' else 1, 
                                 chosen]
            reward = self.rng.rand() < p
            
            q = self.q[q_index,chosen]
            if q == np.inf:  # check for first setting of value
                q = reward
            else:
                q = q + alpha * (reward-q)
            self.q[q_index, chosen] = q
            self.state = 'S0'

    
    
    def softmax(self, values):
        return np.argmax(values + np.random.normal(size=values.shape)*choice_noise)
        
        
        
env = Environment(vocab, seed=2)
        
model = nengo.Network()
with model:
    env_node = nengo.Node(env.node_function, size_in=1)     
        
        
    state_and_action = nengo.Ensemble(n_neurons=N_state_action, dimensions=D*2)      
    nengo.Connection(env_node[:D*2], state_and_action)
    
    prod = nengo.networks.Product(n_neurons=200, dimensions=D)
    transform = np.array([vocab.parse('S0').v,
                          vocab.parse('SA').v,
                          vocab.parse('SB').v,])
    nengo.Connection(env_node[-3:], prod.A, transform=transform.T)
    
    def ideal_transition(x):
        sim_s = np.dot(x[:D], vocab.vectors)
        index_s = np.argmax(sim_s)
        s = vocab.keys[index_s]
        
        sim_a = np.dot(x[D:], vocab.vectors)
        index_a = np.argmax(sim_a)
        a = vocab.keys[index_a]
        
        threshold = 0.1
        
        if sim_s[index_s]<threshold:
            return np.zeros(D)
        if sim_a[index_a]<threshold:
            return np.zeros(D)
        if s == 'S0':
            if a == 'L':
                p = [0,0.7,0.3]
            elif a == 'R':
                p = [0,0.3,0.7]
            else:
                p = [0,0,0]
        elif s == 'SA' or s=='SB':
            p = [1,0,0]
        else:
            p = [0,0,0]
            
        return np.dot(transform.T, p)
    nengo.Connection(state_and_action, prod.B, function=ideal_transition)
    
    nengo.Connection(prod.output, env_node, transform=np.ones((1, D)))    
        