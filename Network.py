from Neuron import *
import random as rand
import numpy as np


class Network:
    def __init__(self, N, p,beta=0):
        self.neurons = []
        self.errors = []
        self.oldState = []
        self.newState = []
        self.orderParam = 0
        rand.seed(rand.random())
        for i in range(N):
            neuron = Neuron(i, N, beta)
            self.neurons.append(neuron)
        for i in range(len(self.neurons)):
            self.neurons[i].connectHopfield(self.neurons)

    def storePatterns(self, patterns, calcW=True):
        for i in range(patterns.shape[0]):
            for j in range(len(self.neurons)):
                self.neurons[j].storePattern(patterns[i])
        if calcW:
            for i in range(len(self.neurons)):
                self.neurons[i].calcWeights()

    def runHopfield(self, patterns):
        r = rand.randint(0, patterns.shape[0] - 1)
        for i in range(len(self.neurons)):
            self.neurons[i].state = patterns[r, i]
        for i in range(len(self.neurons)):
            self.errors.append(self.neurons[rand.randint(0, len(self.neurons) - 1)].singleStep(r))

    def inputPattern(self, input):
        for i in range(len(self.neurons)):
            self.neurons[i].state = input[i]

    def getCurrentNetworkState(self):
        netState = []
        for i in range(len(self.neurons)):
            netState.append(self.neurons[i].state)
        return np.array(netState)

    def distortPattern(self, q, pattern):
        workPattern = np.array(pattern)
        flippedBits = int(q * len(pattern))
        bitPosition = np.random.choice(len(pattern), flippedBits, replace=False)
        workPattern[bitPosition] = -pattern[bitPosition]
        return workPattern

    def runDigits(self, pattern, origPattern):
        self.inputPattern(pattern)
        self.oldState = self.getCurrentNetworkState()
        indexes = np.random.permutation(len(self.neurons))
        for number in indexes:
            self.neurons[number].singleStep(synchronus=False)
        self.newState = self.getCurrentNetworkState()
        if np.all(self.oldState - self.newState == np.zeros(len(self.oldState))):
            if np.all(origPattern - self.newState == np.zeros(len(self.oldState))):
                return 1  # found pattern
            elif np.all(origPattern - self.newState == 2 * origPattern):
                return 2  # found inverted pattern
            else:
                return 3  # found false stable pattern
        self.oldState = self.newState
        return 4  # did not found stable pattern

    def runStochastic(self, pattern):
        self.inputPattern(pattern)
        indexes = np.random.permutation(len(self.neurons))
        for number in indexes:
            self.neurons[number].singleStep(synchronus=False,deterministic=False)
        self.orderParam=self.calcMMu(1)


    def calcMMu(self,mu):
        sumResult=0
        for  thisNeuron in self.neurons:
            sumResult+=thisNeuron.storedPatterns[mu-1]*np.mean(thisNeuron.prevStates)
        mmu=sumResult*1.0/len(self.neurons)
        return mmu
