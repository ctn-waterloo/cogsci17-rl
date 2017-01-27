import random
import numpy as np

# Uses the Q-update strategy found in Daw et al. 2011 supplemental materials

# This one learns the transition probabitilies

class ModelBasedForward:
    def __init__(self, actions, states, epsilon=0.1, alpha=0.2, eta = 0.05, noise = 0.05):
        self.q={} # this is a dictionary of the form: 
                # key: (state, action)
                # value: q_value 
        self.model_P = {} # this is a dictionary of the form:
                    # key: (state, action)
                    # value: [state0_count, state1_count, state2_count]
        # may not need this, if we just use the actual reward that was received 
        self.model_R = {} # this is a dictionary of the form:
            # key: (state, action)
            # value: reward
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta
        self.noise = noise
        self.actions = actions
        self.states = states
        self.prev = None
        self.terminal_level = len(self.states)
        self.initializeTransitionProbabilities()

    #
    def initializeTransitionProbabilities(self):
        for level in self.states.keys():
            next_level = level+1
            if level == self.terminal_level:
                next_level = 1
                # I'm assuming I can safely assume that there will always be only one state in the first level!!!!!!!
                # This could be easily changed if this assumption ever turned out to be wrong...
                for state1 in self.states[level]:
                    for action in self.actions:
                        for state2 in self.states[next_level]:
                            self.model_P[(state1, action, state2)] = 1.0
            else:
                for state in self.states[level]:
                    num_next_states = len(self.states[next_level])
                    for action in self.actions:
                        for next_state in self.states[next_level]:
                            self.model_P[(state, action, next_state)]=1.0/num_next_states


    # Okay, I'm going to decide that the code for a particular task is going to
    # keep track of which level it's on
    # From Glascher et al. 2010
    
    def updateTransitionProbabilities(self, state1, action, state2, state2_level):
        error = 1 - self.model_P[(state1, action, state2)]
        self.model_P[(state1, action, state2)] += self.eta*error
        # now we have to reduce the probabilities of all states not arrived in
        for level_state in self.states[state2_level]:
            if level_state != state2:
                self.model_P[(state1, action, level_state)] *= (1-self.eta)
    """

    # From Akam et al. 2015
    def updateTransitionProbabilities(self, state1, action, state2, state2_level):
        self.model_P[(state1, action, state2)] = (1-self.eta)*self.model_P[(state1, action, state2)] + self.eta
        # now we have to reduce the probabilities of all states not arrived in
        for level_state in self.states[state2_level]:
            if level_state != state2:
                self.model_P[(state1, action, level_state)] *= (1-self.eta)
"""




    def getQ(self, state, action):
        return self.q.get((state,action),0.0)


    def learnQ(self, state, action, value):
        oldv = self.q.get((state, action), None)
        if oldv == None:
            self.q[(state,action)]=value
        else:
            #self.q[(state,action)]=oldv+self.alpha*(value-oldv)
            self.q[(state, action)]= (1-self.alpha)*oldv + self.alpha*value
            #print (state, action), ':', oldv, '->', value
    

    def learn(self, state1, action, reward, state2, state2_level):
        self.updateTransitionProbabilities(state1, action, state2, state2_level)
        if state1 == 0:
            assert reward==0

        if state2 not in self.states[state2_level]:
            print "State " + str(state2) + " is not in level " + str(state2_level)
            return None
        
        # "state1" is state 0
        if state2_level == 2: # we are currently learning the value of the first stage (state 0)
            # recompute the Q-values for each possible action based on 
            # current estimates of transition probabilities and rewards at stage 2
            for a in self.actions:
                self.q[(state1, a)] = self.calc_value(state1, a, state2_level) 

        # "state1" is either state 1 or 2
        elif state2_level == 1: # we are currently learning the value of the second state (state 1 or 2)
            #qnext = self.calc_value(state1, action, state2_level)
            self.learnQ(state1, action, reward)
         
        return 1

    def learn_nengo(self, state1, action, reward, state2, state2_level,
                    value_nengo):
        self.updateTransitionProbabilities(state1, action, state2, state2_level)
        if state1 == 0:
            assert reward==0

        if state2 not in self.states[state2_level]:
            print "State " + str(state2) + " is not in level " + str(state2_level)
            return None
        
        # "state1" is state 0
        if state2_level == 2: # we are currently learning the value of the first stage (state 0)
            # recompute the Q-values for each possible action based on 
            # current estimates of transition probabilities and rewards at stage 2
            for i, a in enumerate(self.actions):
                self.q[(state1, a)] = value_nengo[i]
                #self.q[(state1, a)] = self.calc_value(state1, a, state2_level)
                #print(a, self.q[(state1, a)])
                #print(a, self.calc_value(state1, a, state2_level))
                #print(a, state1)
                #self.q[(state1, a)] = reward_nengo #THIS IS WRONG!!!!!!!!!!!!! #also put all ens in direct mode except learning one for now, to speed things up
            #print("")
        # "state1" is either state 1 or 2
        elif state2_level == 1: # we are currently learning the value of the second state (state 1 or 2)
            #qnext = self.calc_value(state1, action, state2_level)
            self.learnQ(state1, action, reward)
         
        return 1

    # this will be replaced by the neural part eventually
    # state2_level could possibly be replaced by just all the states
    def calc_value(self, state1, action, state2_level):
        value = 0
        for state2 in self.states[state2_level]:
            next_action = self.max_action(state2)
            temp_value = self.model_P[(state1, action, state2)] * self.getQ(state2, next_action)
            value += temp_value
        return value

    # Choose A from S using policy derived from Q 
    def chooseAction(self,state):
        #if random.random()<self.epsilon:
        #    return random.choice(self.actions)
        #else:
        #    return self.max_action(state)

        
        q=[self.getQ(state,a)+random.normalvariate(0, self.noise) for a in self.actions]
        i=q.index(max(q))
        action=self.actions[i]
        return action

    # returns the action with the highest Q-value
    def max_action(self, state):
        q=[self.getQ(state,a) for a in self.actions]

        maxQ=max(q)
        count=q.count(maxQ)
        if count>1:
            best=[i for i in range(len(self.actions)) if q[i]==maxQ]
            i=random.choice(best)
        else:
            i=q.index(maxQ)
        action=self.actions[i]
        return action

    










    # update the model?
    # this is not the best way to do this, but it's a place to start
    #def updateModelProbabilities(self, state1, action, state2):
    # default 1.0 because if we've only seen it once, we think 
    # this transition is 100% likely
    #    oldp = self.model_P.get((state1, action), [0, 0, 0])
    #    oldp[state2] += 1
    #    self.model_P[(state1, action)]=oldp

    # basically, I'm treating the state transition probability
    # as the proportion of transitions to state2 (given the action)
    # to the total number of transitions from state1 (given the action)
    #def getStateProbability(self, state1, action, state2):
    #    counts = self.model_P.get((state1, action), [0, 0, 0])
    #    total = sum(counts)
    #    prob = 0
    #    if total > 0:
    #        prob = counts[state2]/total
    #    return prob
