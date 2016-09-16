import sys
import numpy as NP


class Neuron:
    def __init__(self, ID, N):
        self.ID = ID
        self.N = N
        self.inputs = []  # list of connected Neurons
        self.weights = []  # list of weights of the Neurons
        self.storedPatterns = []
        self.state = 0
        self.nextState = 0

    def connectHopfield(self, Neurons):
        for i in range(len(Neurons)):
            if Neurons[i].ID != self.ID:
                self.inputs.append(Neurons[i])

    def storePattern(self, pattern):
        if len(pattern) == self.N:
            self.storedPatterns.append(pattern[self.ID])
        else:
            sys.exit("stored pattern has an incorrect size")

    def calcWeights(self):
        for i in range(len(self.inputs)):
            sumResult = 0
            for j in range(len(self.storedPatterns)):
                sumResult += self.inputs[i].storedPatterns[j] * self.storedPatterns[j]
            w = sumResult * 1.0 / self.N
            self.weights.append(w)

    def singleStep(self):
        sumResult = 0
        for i in range(len(self.inputs)):
            sumResult += self.weights[i] * self.inputs[i].state
        self.nextState = NP.sign(sumResult)

    def transferStates(self):
        self.state = self.nextState
