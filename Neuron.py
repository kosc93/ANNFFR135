import sys
import numpy as NP

class Neuron:
    ID = 0 # identifies the neuron
    inputs = [] # list of connected Neurons
    weights = [] # list of weights of the Neurons
    N = 0 # total number of Neurons
    storedPatterns = []
    state = 0

    def __init__(self, ID, N):
        self.ID = ID
        self.N = N


    def connectHopfield(self , Neurons):
        for i in range(len(Neurons)):
            if Neurons[i].ID != self.ID:
                self.inputs.append(i)

    def storePattern(self, pattern):
        if len(pattern)==self.N:
            self.storedPatterns.append(pattern[self.ID])
        else:
          sys.exit("stored pattern has an incorrect size")

    def calcWeights(self):
        for i in range(len(self.inputs)):
            sumResult=0
            for j in range(len(self.storedPatterns)):
                sumResult+=self.inputs[i].storedPattern[j]*self.storedPatterns[j]
            w=1/self.N*sumResult
            self.weights.append(w)

    def singleStep(self):
        sumResult=0
        for i in range(len(self.inputs)):
            sumResult+=self.weights[i]* self.inputs[i].state
        self.state=NP.sign(sumResult)



