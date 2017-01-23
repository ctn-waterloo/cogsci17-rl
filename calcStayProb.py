from __future__ import division
from __future__ import print_function  # Only needed for Python 2

# A probably overly complicated way to determine the probability of repeating the action of stage 1, trial n in stage 1, trial n+1
# Should produce numbers that look like Fig. 2 of Daw et al. 2011
# Input should be in the form of: firstStageChoice, secondStageState, secondStageChoice, reward

#### This is incomplete #####
# How did I get it working before? I may have just hardcoded the filename

class CalcStayProb():
    def __init__(self, actions=["left", "right"], states = [1, 2]):
        self.actions = actions
        self.numActions = len(actions)
        self.trr = 0 # total rare rewarded
        self.tru = 0 # total rare unrewarded
        self.tcr = 0 # total common rewarded
        self.tcu = 0 # total common unrewarded
        self.trr_s = 0 # total rare rewarded stays
        self.tru_s = 0 # total rare unrewarded stays
        self.tcr_s = 0 # total common rewarded stays
        self.tcu_s = 0 # total common unrewarded stays
        self.states = states
        self.numStates = len(states)


    def countFile(self, f):
        first = f.readline()
        second = f.readline()
        while second != "":
            firstChoice, nextState, secondChoice, reward = first.split(' ')
            reward = int(reward)
            nextFirst = second.split(' ')[0]
            # okay, this is not general at all, but I'm going to program it this way first so that I know roughly what I'm doing
            if firstChoice == "left":
                #print "a"
                if nextState == "2":
                    #print "b"
                    if reward == 1:
                        #print "c"
                        # rare rewarded case
                        self.trr += 1
                        if nextFirst == firstChoice:
                            #print "d"
                            self.trr_s += 1
                    elif reward == 0:
                        #print "e"
                        # rare unrewarded case
                        self.tru += 1
                        if nextFirst == firstChoice:
                            #print "f"
                            self.tru_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 1
                elif nextState == "1":
                    #print "g"
                    if reward == 1:
                        #print "h"
                        # common rewarded case
                        self.tcr += 1
                        if nextFirst == firstChoice:
                            #print "i"
                            self.tcr_s += 1
                    elif reward == 0:
                        #print "j"
                        # common unrewarded case
                        self.tcu += 1
                        if nextFirst == firstChoice:
                            #print "k"
                            self.tcu_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 2
                else:
                    print ("something may have gone wrong with parsing the second token")
                    return 3
            elif firstChoice == "right":
                #print "l"
                if nextState == "1":
                    #print "m"
                    if reward == 1:
                        #print "n"
                        #rare rewarded case
                        self.trr += 1
                        if nextFirst == firstChoice:
                            #print "o"
                            self.trr_s += 1
                    elif reward == 0:
                        #print "p"
                        # rare unrewarded case
                        self.tru += 1
                        if nextFirst == firstChoice:
                            #print "q"
                            self.tru_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 4
                elif nextState == "2":
                    #print "r"
                    if reward == 1:
                        #print "s"
                        # common rewarded case
                        self.tcr += 1
                        if nextFirst == firstChoice:
                            #print "t"
                            self.tcr_s += 1
                    elif reward == 0:
                        #print "u"
                        # common unrewarded case
                        self.tcu += 1
                        if nextFirst == firstChoice:
                            #print "v"
                            self.tcu_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 5
                else:
                    print ("something may have gone wrong with parsing the second token")
                    return 6
            else:
                print ("something may have gone wrong with parsing the first token")
                return 7
            first = second
            second = f.readline()
        # everything probably worked!
        return 0
    
    def countStrings(self, f):
        first = f[0]
        second = f[1]
        i = 1
        while i < len(f):
            firstChoice, nextState, secondChoice, reward = first.split(' ')
            reward = int(reward)
            nextFirst = second.split(' ')[0]
            # okay, this is not general at all, but I'm going to program it this way first so that I know roughly what I'm doing
            if firstChoice == "left":
                #print "a"
                if nextState == "2":
                    #print "b"
                    if reward == 1:
                        #print "c"
                        # rare rewarded case
                        self.trr += 1
                        if nextFirst == firstChoice:
                            #print "d"
                            self.trr_s += 1
                    elif reward == 0:
                        #print "e"
                        # rare unrewarded case
                        self.tru += 1
                        if nextFirst == firstChoice:
                            #print "f"
                            self.tru_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 1
                elif nextState == "1":
                    #print "g"
                    if reward == 1:
                        #print "h"
                        # common rewarded case
                        self.tcr += 1
                        if nextFirst == firstChoice:
                            #print "i"
                            self.tcr_s += 1
                    elif reward == 0:
                        #print "j"
                        # common unrewarded case
                        self.tcu += 1
                        if nextFirst == firstChoice:
                            #print "k"
                            self.tcu_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 2
                else:
                    print ("something may have gone wrong with parsing the second token")
                    return 3
            elif firstChoice == "right":
                #print "l"
                if nextState == "1":
                    #print "m"
                    if reward == 1:
                        #print "n"
                        #rare rewarded case
                        self.trr += 1
                        if nextFirst == firstChoice:
                            #print "o"
                            self.trr_s += 1
                    elif reward == 0:
                        #print "p"
                        # rare unrewarded case
                        self.tru += 1
                        if nextFirst == firstChoice:
                            #print "q"
                            self.tru_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 4
                elif nextState == "2":
                    #print "r"
                    if reward == 1:
                        #print "s"
                        # common rewarded case
                        self.tcr += 1
                        if nextFirst == firstChoice:
                            #print "t"
                            self.tcr_s += 1
                    elif reward == 0:
                        #print "u"
                        # common unrewarded case
                        self.tcu += 1
                        if nextFirst == firstChoice:
                            #print "v"
                            self.tcu_s += 1
                    else:
                        print ("something probably wrong with your string parsing of last token")
                        return 5
                else:
                    print ("something may have gone wrong with parsing the second token")
                    return 6
            else:
                print ("something may have gone wrong with parsing the first token")
                return 7
            i += 1
            if i == len(f):
                break
            first = second
            second = f[i]
        # everything probably worked!
        return 0

    def calcStayProb(self, outfile=None, return_value=False):
        stay_tcr = self.tcr_s/self.tcr
        stay_trr = self.trr_s/self.trr
        stay_tcu = self.tcu_s/self.tcu
        stay_tru = self.tru_s/self.tru
        if outfile is not None:
            print('{0} {1} {2} {3}'.format(stay_tcr, stay_trr, stay_tcu, stay_tru), file=outfile)
        elif return_value:
            return (stay_tcr, stay_trr, stay_tcu, stay_tru)
        else:
            print(stay_tcr, stay_trr, stay_tcu, stay_tru)

    def doItAll(self, fileName, outfile=None):
        f = open(fileName)
        # maybe some other stuff here to catch exceptions
        success = self.countFile(f)
        if success == 0:
            # good
            self.calcStayProb(outfile=outfile)
        else:
            print(success)
    
    def doItAllString(self, strings, outfile=None, return_value=False):
        # maybe some other stuff here to catch exceptions
        success = self.countStrings(strings)
        if success == 0:
            # good
            if return_value:
                return self.calcStayProb(outfile=outfile, return_value=True)
            else:
                self.calcStayProb(outfile=outfile)
        else:
            print(success)


if __name__ == "__main__":
    # something here to parse the input string for the correct file name
    calculator = CalcStayProb()
    calculator.doItAll("temp_file.txt")
