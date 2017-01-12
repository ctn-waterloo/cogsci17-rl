from __future__ import division

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
						print "something probably wrong with your string parsing of last token"
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
						print "something probably wrong with your string parsing of last token"
						return 2
				else:
					print "something may have gone wrong with parsing the second token"
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
						print "something probably wrong with your string parsing of last token"
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
						print "something probably wrong with your string parsing of last token"
						return 5
				else:
					print "something may have gone wrong with parsing the second token"
					return 6
			else:
				print "something may have gone wrong with parsing the first token"
				return 7
			first = second
			second = f.readline()
		# everything probably worked!
		return 0

	def calcStayProb(self):
		stay_tcr = self.tcr_s/self.tcr
		stay_trr = self.trr_s/self.trr
		stay_tcu = self.tcu_s/self.tcu
		stay_tru = self.tru_s/self.tru
		print stay_tcr, stay_trr, stay_tcu, stay_tru

	def doItAll(self, fileName):
		f = open(fileName)
		# maybe some other stuff here to catch exceptions
		success = self.countFile(f)
		if success == 0:
			# good
			self.calcStayProb()
		else:
			print success


if __name__ == "__main__":
	# something here to parse the input string for the correct file name
	calculator = CalcStayProb()
	calculator.doItAll("trial_6.txt")