from __future__ import print_function  # Only needed for Python 2
import random
import modelbasedforward as mb
import numpy as np

# Uses the Q-update strategy found in Daw et al. 2011 supplemental materials
# Implements the Daw task without learning the transition probabilities 

class Agent():
    def __init__(self, vocab, time_interval, q_scaling=1, outfile=None, randomReward = True, case1 = 0.25, case2 = 0.5, case3 = 0.5, case4 = 0.75):
        state_dict = {1:[0], 2:[1, 2]}
        transition_dict = {(0, "left", 1):0.7,
                            (0, "left", 2):0.3,
                            (0, "right", 1):0.3,
                            (0, "right", 2):0.7,
                            (1, "left", 0):1.0,
                            (1, "right", 0):1.0,
                            (2, "left", 0):1.0,
                            (2, "right", 0):1.0}
        self.ai = mb.ModelBasedForward(actions=["left", "right"], states = state_dict, transitions=transition_dict)
        self.lastAction = None
        self.lastState = None
        self.numLevels = len(state_dict)
        self.firstLevel = 1
        self.currBoardState = 0
        self.lastBoardState = None
        self.currLevel = 1
        self.currAction = None
        self.pastReward = 0
        self.currReward = 0
        # stuff to set up the random walk of reward
        self.SD = 0.025
        self.lowerBoundary = 0.25
        self.upperBoundary = 0.75
        if randomReward:
            self.case1RewardProb = self.initializeReward()
            self.case2RewardProb = self.initializeReward()
            self.case3RewardProb = self.initializeReward()
            self.case4RewardProb = self.initializeReward()
        else:
            self.case1RewardProb = case1
            self.case2RewardProb = case2
            self.case3RewardProb = case3
            self.case4RewardProb = case4


        # Saving data to a file
        self.outfile = outfile
        self.firstStageChoice = None
        self.secondStage = None
        self.secondStageChoice = None
        self.finalReward = None
        self.step = 0
        self.result_string = []
        # Nengo stuff
        self.q_scaling = q_scaling
        self.time_interval = time_interval
        self.states = ['S0', 'S1', 'S2']
        self.actions = ['L', 'R']
        self.action_strings = ['left', 'right']
        self.vocab = vocab
        self.dim = len(self.vocab.vectors[0]) # Get the dimensionality of the vocab
        self.action_vec = np.zeros((self.dim))
        self.state_vec = np.zeros((self.dim))
        self.q_vec = np.zeros((self.dim))

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

    def initializeReward(self):
        rewardProb = 0
        while rewardProb < self.lowerBoundary or rewardProb > self.upperBoundary:
            rewardProb = random.random()
        return rewardProb 

    def getLastBoardState(self):
        return self.lastBoardState

    def getCurrBoardState(self):
        return self.currBoardState

    def getLastAction(self):
        return self.lastAction

    def getCurrReward(self):
        return self.currReward

    # random walk function
    def randomWalk(self, oldValue):
        newValue = 0
        noise = random.gauss(0, self.SD)
        addNoise = oldValue + noise
        if addNoise > self.upperBoundary:
            diff = self.upperBoundary - oldValue # how much distance between old value and upper boundary
            extra = noise - diff # how much the noise makes the value go over the upper boundary
            pointDiff = diff - extra # reflecting back, should be pos if 
            newValue = oldValue + pointDiff # old value plus whatever reflecting value we've calculated
        elif addNoise < self.lowerBoundary:
            diff = oldValue - self.lowerBoundary
            extra = -noise - diff
            pointDiff = diff - extra
            newValue = oldValue - pointDiff
        else:
            newValue = addNoise
        return newValue

    # this should return the current reward based on the 
    # action taken in the current state
    def calcReward(self, currState, currAction):
        if currState == 0:
            return 0
        currProb = random.random()  
        if currState == 1:
            if currAction == "left": # choose left (CASE 1)
                reward = 0 
                #print currProb, self.case1RewardProb
                if currProb > self.case1RewardProb:
                    reward = 1
                return reward
            elif currAction == "right": # choose right (CASE 2)
                reward = 0 
                #print currProb, self.case2RewardProb
                if currProb > self.case2RewardProb:
                    reward = 1
                return reward
            else:
                print("Something went very wrong with choosing the action: should be either left or right")
                return None
        if currState == 2:
            if currAction == "left": # choose left (CASE 3)
                reward = 0 
                #print currProb, self.case3RewardProb
                if currProb > self.case3RewardProb:
                    reward = 1
                return reward
            elif currAction == "right": # choose right (CASE 4) 
                reward = 0 
                #print currProb, self.case4RewardProb
                if currProb > self.case4RewardProb:
                    reward = 1
                return reward
            else:
                print("Something went very wrong with choosing the action: should be either left or right")
                return None

    def updateRewardProb(self):
        self.case1RewardProb = self.randomWalk(self.case1RewardProb)
        self.case2RewardProb = self.randomWalk(self.case2RewardProb)
        self.case3RewardProb = self.randomWalk(self.case3RewardProb)
        self.case4RewardProb = self.randomWalk(self.case4RewardProb)

    # calculates the next state probabilistically
    # (may want to include some way to change these probabilities externally)
    # the paper does say that this prob was fixed throughout the experiment
    def calcNextState(self, currState, currAction):
        nextState = 0
        if currState == 0:
            #print "here"
            if currAction == "left":
                state1Prob = random.random()
                if state1Prob > 0.3: # more likely to be state 1
                    nextState = 1
                else:
                    nextState = 2
            if currAction == "right":
                state1Prob = random.random()
                if state1Prob > 0.7: # more likely to be state 2
                    nextState = 1
                else:
                    nextState = 2
        return nextState

    def calcNextLevel(self):
        if self.currLevel == self.numLevels:
            return self.firstLevel
        else:
            return self.currLevel + 1

    def oneStep(self, reward_nengo):
        #print ""
        #print "debug:"
        #print "    ", self.lastBoardState, self.lastAction, self.currBoardState, self.currLevel
        currAction = self.ai.chooseAction(self.currBoardState)
        #print "  and the current action is", currAction
        nextBoardState = self.calcNextState(self.currBoardState, currAction)
        self.currReward = self.calcReward(self.currBoardState, currAction)
        self.updateRewardProb() #bookkeeping step
        
        if self.lastAction != None:
            #print "  learning is happening"
            if self.ai.learn_nengo(self.lastBoardState, self.lastAction, self.pastReward, self.currBoardState, self.currLevel, reward_nengo) == None:
                return None
        # more bookkeeping
        self.lastBoardState = self.currBoardState
        self.currBoardState = nextBoardState
        self.currLevel = self.calcNextLevel()
        self.pastReward = self.currReward
        self.lastAction = currAction
        return 1    
    
    def __call__(self, t, reward_nengo):
        #TODO: filter the reward value over the time interval to get a less noisy result
        if t - self.last_t >= self.time_interval:
            self.oneStep(reward_nengo)
            action = self.lastAction
            state = self.getCurrBoardState()
            q = self.ai.getQ(state, action)
            
            self.q_vec = np.zeros((self.dim))
            for i, s in enumerate(self.states):
                next_action = self.ai.max_action(i)
                q_val = self.ai.getQ(i, next_action)
                self.q_vec += self.index_to_state_vector[state] * q_val * self.q_scaling

            self.action_vec = self.index_to_action_vector[self.action_strings.index(action)]
            self.state_vec = self.index_to_state_vector[state]
            #TODO: make sure this is correct

            self.last_t = t

            if self.step%2 == 0: # in stage 1
                self.firstStageChoice = self.getLastAction()
                self.secondStage = self.getCurrBoardState()
            else: # in stage 2
                self.secondStageChoice = self.getLastAction()
                self.finalReward = self.getCurrReward()
                if self.outfile is not None:
                    # Print results to a file
                    print('{0} {1} {2} {3}'.format(self.firstStageChoice, self.secondStage, self.secondStageChoice, self.finalReward), file=self.outfile)
                else:
                    self.result_string.append('{0} {1} {2} {3}'.format(self.firstStageChoice, self.secondStage, self.secondStageChoice, self.finalReward))
            self.step += 1
        
        return np.concatenate((self.action_vec, self.state_vec, self.q_vec))
