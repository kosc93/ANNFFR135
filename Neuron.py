import sys
import numpy as np


class Neuron:
    def __init__(self, ID, N, beta=0):
        self.ID = ID
        self.N = N
        self.inputs = []  # list of connected Neurons
        self.weights = []  # list of weights of the Neurons
        self.storedPatterns = []
        self.state = 0
        self.nextState = 0
        self.stepError = 0
        self.beta=beta
        if beta>0:
            self.prevStates=[]

    def connectHopfield(self, Neurons):
        for i in range(len(Neurons)):
            self.inputs.append(Neurons[i])

    def storePattern(self, pattern):
        if len(pattern) == self.N:
            self.storedPatterns.append(pattern[self.ID])
        else:
            sys.exit("stored pattern has an incorrect size")

    def calcWeights(self):
        self.weights = []
        for i in range(len(self.inputs)):
            if i == self.ID:
                w = 0
            else:
                sumResult = 0
                for j in range(len(self.storedPatterns)):
                    sumResult += self.inputs[i].storedPatterns[j] * self.storedPatterns[j]
                w = sumResult * 1.0 / self.N
            self.weights.append(w)

    def singleStep(self, pattern=-1, synchronus=True,deterministic=True):
        sumResult = 0
        for i in range(len(self.inputs)):
            sumResult += self.weights[i] * self.inputs[i].state
        if deterministic:
            self.nextState = np.sign(sumResult)
        else:
            res=(0.5-0.5*np.tanh(self.beta*sumResult)) #propability of a -1
            res=np.random.choice(2,1,p=[res,1-res])[0]
            if res==0:
                res=-1
            self.nextState=res
            self.prevStates.append(self.nextState)

        if pattern >= 0:
            return self.calcPerror(pattern)
        if not synchronus:
            self.transferStates()

    def transferStates(self):
        self.state = self.nextState



    def calcPerror(self, pattern):
        sum = 0
        for i in range(len(self.inputs)):
            if i != self.ID:
                for j in range(len(self.storedPatterns)):
                    if pattern != j:
                        sum += self.storedPatterns[j] * self.inputs[i].storedPatterns[j] * self.inputs[j].state
        c = -(self.state * 1.0 / self.N) * sum
        if c > 1:
            return 1
        else:
            return 0
