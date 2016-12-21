import numpy as np

class Environment(object):

    def __init__(self, vocab, time_interval=.5):

        self.states = ['S0', 'S1', 'S2']
        self.actions = ['L', 'R']
        self.vocab = vocab
        self.dim = len(self.vocab.vectors[0]) # Get the dimensionality of the vocab

        # TODO: these two mappings seems useless, remove them if they are
        # dictionary mapping action string to index in the transition matrix
        self.action_to_index = {}

        # dictionary mapping state string to index in the transition matrix
        self.state_to_index = {}
        
        # takes a state index and returns the corresponding vector for the semantic pointer
        self.index_to_state_vector = np.zeros((len(self.states), self.dim))

        # takes an action index and returns the corresponding vector for the semantic pointer
        self.index_to_action_vector = np.zeros((len(self.actions), self.dim))

        # The last time a state transition was made
        # This is used to make sure a set amount of time goes by before another transition happens
        self.last_t = 0

        # The amount of time before another state transition can be made
        self.time_interval = time_interval

        # Fill in mapping data structures based on the vocab given
        for i, vk in enumerate(self.vocab.keys):
            if vk in self.actions:
                self.action_to_index[vk] = self.actions.index(vk)
                self.index_to_action_vector[self.actions.index(vk)] = self.vocab.vectors[i]

            if vk in self.states:
                self.state_to_index[vk] = self.states.index(vk)
                self.index_to_state_vector[self.states.index(vk)] = self.vocab.vectors[i]


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
        for i, p in enumerate(prob):
            if rn < p:
                self.current_state = i
                break

    def find_closest_action(self, vec):
        # Find the dot product with all other vectors
        distance = 0
        best_index = 0
        for i, v in enumerate(self.index_to_action_vector):
            d = np.dot(vec, v)
            if d > distance:
                distance = d
                best_index = i

        # Return the index of the maximum dot product (closest vector)
        return best_index

        #TODO: there should be a nice way to vectorize this
        """
        # Find the dot product with all other vectors
        distance = np.dot(vec, self.index_to_action_vector)

        # Return the index of the maximum dot product (closest vector)
        return np.argmax(distance)
        """

    def __call__(self, t, x):
        """
        Takes in a semantic pointer representing the action to take
        Returns a semantic pointer representing the resulting next state
        """
        #TODO: might want to integrate/average over the input so it isn't only based on the last timestep

        # Only perform a state transition if enough time has passed since the last one
        if t - self.last_t >= self.time_interval:
            # Get the action id of the closest action vector to the noisy input vector
            action = self.find_closest_action(x)
            # Perform that action to update the current state
            self.make_action(action)

            self.last_t = t

        return self.index_to_state_vector[self.current_state]

