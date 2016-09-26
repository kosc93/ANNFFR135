import sys
import numpy as np


class Neuron:
    def __init__(self, ID, N, beta=0, bias=0, eta=0):
        self.ID = ID
        self.N = N
        self.inputs = np.array([])  # list of connected Neurons
        self.weights = np.array([])  # list of weights of the Neurons
        self.storedPatterns = []
        self.state = 0
        self.rawOutput = 0
        self.nextState = 0
        self.stepError = 0
        self.beta=beta
        self.bias=bias
        self.eta=eta
        if beta>0:
            self.prevStates=[]

    def connect(self, Neurons):
        for neuron in Neurons:
            self.inputs=np.append(self.inputs,neuron)


    def storePattern(self, pattern):
        if len(pattern) == self.N:
            self.storedPatterns.append(pattern[self.ID])
        else:
            sys.exit("stored pattern has an incorrect size")

    def calcWeights(self):
        self.weights=np.array([])
        for i in range(len(self.inputs)):
            if i == self.ID:
                w = 0
            else:
                sumResult = 0
                for j in range(len(self.storedPatterns)):
                    sumResult += self.inputs[i].storedPatterns[j] * self.storedPatterns[j]
                w = sumResult * 1.0 / self.N
            self.weights=np.append(self.weights,w)


    def singleStep(self, pattern=-1, synchronus=True,deterministic=True):
        sumResult = 0
        for i in range(len(self.inputs)):
            sumResult += self.weights[i] * self.inputs[i].state
        if deterministic:
            self.nextState = np.sign(sumResult-self.bias)
        else:
            self.rawOutput=np.tanh(self.beta*sumResult-self.bias)
            res=(0.5-0.5*self.rawOutput) #propability of a -1
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

    def updateFF(self, desiredState,output=[],hidden=[]):
        if not hidden:
            #update biases and weights of  Output without hidden layer
            pass
        elif not output:
            #update biases and weights of  Output with hidden layer
            pass
        else:
            #update update biases and weights of  hidden layer
            pass