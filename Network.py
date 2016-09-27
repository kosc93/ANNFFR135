from Neuron import *
import random as rand
import numpy as np


class Network:
    def __init__(self, N,beta=0,Input=[],Output=[],Hidden=[]):
        self.neurons = np.array([])
        self.errors = []
        self.oldState = []
        self.newState = []
        self.inputs=np.array([])
        self.hiddens=np.array([])
        self.outputs=np.array([])
        rand.seed(rand.random())
        for i in range(N):
            neuron = Neuron(i, N, beta)
            self.neurons=np.append(self.neurons,neuron)
        if len(Input)==0:
            for neuron in self.neurons:
                neuron.connect(self.neurons)
            self.inputs=self.neurons
            self.outputs=self.neurons
        else:
            self.inputs=self.neurons[Input]
            if len(Hidden)==0:
                self.outputs=self.neurons[Output]
                for neuron in self.outputs:
                    neuron.connect(self.neurons[Input])
            else:
                self.outputs=self.neurons[Output]
                for neuron in self.outputs:
                    neuron.connect(self.neurons[Hidden])
                self.hiddens=self.neurons[Hidden]
                for neuron in self.hiddens:
                    neuron.connect(self.neurons[Input])
            self.initRandom()

    def storePatterns(self, patterns, calcW=True):
        for pattern in patterns:
            for neuron in self.neurons:
                neuron.storePattern(pattern)
        if calcW:
            for neuron in self.neurons:
                neuron.calcWeights()

    def runHopfield(self, patterns):
        r = rand.randint(0, patterns.shape[0] - 1)
        for i in range(len(self.neurons)):
            self.neurons[i].state = patterns[r, i]
        for i in range(len(self.neurons)):
            self.errors.append(self.neurons[rand.randint(0, len(self.neurons) - 1)].singleStep(r))

    def inputPattern(self, input):
        for i,neuron in enumerate(self.inputs):
                neuron.state = input[i]

    def getCurrentNetworkState(self):
        netState = np.array([])
        for neuron in self.outputs:
            netState=np.append(netState,neuron.state)
        return netState

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

    def calcMMu(self,mu):
        sumResult=0
        for  thisNeuron in self.neurons:
            sumResult+=thisNeuron.storedPatterns[mu-1]*np.mean(thisNeuron.prevStates)
        mmu=sumResult*1.0/len(self.neurons)
        return mmu

    def initRandom(self):
        for neuron in self.neurons:
            neuron.bias=np.random.uniform(-1.0,1.0)
            for input in neuron.inputs:
                neuron.weights=np.append(neuron.weights,np.random.uniform(-0.2,0.2))

    def trainFF(self, pattern, desiredState):
        self.inputPattern(pattern)
        runNeurons=np.append(self.hiddens,self.outputs)
        for neuron in runNeurons:
                neuron.singleStep(synchronus=False,deterministic=False)
        for neuron in self.outputs:
            neuron.updateFF(desiredState,hidden=self.hiddens)
        for neuron in self.hiddens:
            neuron.updateFF(desiredState,output=self.outputs,input=self.inputs)

    def calcError(self,zetas=[],xis=[]):
        sumRes=0
        for i,xi in enumerate(xis):
            self.inputPattern(xi)
            runNeurons = np.append(self.hiddens, self.outputs)
            for neuron in runNeurons:
                neuron.singleStep(synchronus=False, deterministic=False)
            res=zetas[i]-np.sign(np.tanh(self.outputs[0].beta*self.outputs[0].rawOutput))
            sumRes+=np.abs(res)
        return sumRes/2.0*len(xis)
