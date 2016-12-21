import numpy as np

class Environment(object):

    def __init__(self, vocab, time_interval=10):

        self.states = ['S0', 'S1', 'S2']
        self.actions = ['L', 'R']
        self.vocab = vocab
        self.action_to_index = {} # dictionary mapping action string to index in the transition matrix

        # The last time a state transition was made
        # This is used to make sure a set amount of time goes by before another transition happens
        self.last_t = 0

        # The amount of time before another state transition can be made
        self.time_interval = time_interval

        for vk in self.vocab.keys:
            if vk in self.actions:
                self.action_to_index[vk] = self.actions.index(vk)


        # Transition probabilities for state-action pairs
        #transition_probabilities = [[[0, .7, .3],[0, .3, .7]],
        #                    [[1, 0, 0],[1, 0, 0]],
        #                    [[1, 0, 0],[1, 0, 0]]]
        # CDF version of transition probabilities
        transition_probabilities = [[[0, .7, 1],[0, .3, 1]],
                    [[1, 1, 1],[1, 1, 1]],
                    [[1, 1, 1],[1, 1, 1]]]
        #self.transitions = np.zeros((3,2,3))
        self.transitions = np.array(transition_probabilities)

        # Index of the current state
        self.current_state = 0

    def make_action(self, action):
        prob = self.transitions[self.current_state, action]
        rn = np.random.random_sample()
        for i in prob:
            if rn < i:
                self.current_state = i
                break

    def __call__(self, t, x):
        """
        Takes in a semantic pointer representing the action to take
        Returns a semantic pointer representing the resulting next state
        """

        # Only perform a state transition if enough time has passed since the last one
        if t - self.last_t >= self.time_interval:
            pass

        return self.index_to_state_vector[self.current_state]

