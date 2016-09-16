from Neuron import *
import random as rand
import numpy as np
import matplotlib.pyplot as plot


# todo calculate weightmatrix in network and store it in neurons for more flexibility


class Network:
    neurons = []

    def __init__(self, N, p):
        for i in range(N):
            self.neurons.append(Neuron(i, N))
        for i in range(len(self.neurons)):
            self.neurons[i].connectHopfield(self.neurons)

    def storePatterns(self, patterns):
        for i in range(patterns.shape[0]):
            for j in range(len(self.neurons)):
                self.neurons[j].storePattern(patterns[i])
        for i in range(len(self.neurons)):
            self.neurons[i].calcWeights()

    def runHopfield(self, patterns):
        r = rand.randint(0, patterns.shape[0] - 1)
        for i in range(len(self.neurons)):
            self.neurons[i].state = patterns[r, i]
        for i in range(len(self.neurons)):
            self.neurons[i].singleStep()


if __name__ == '__main__':
    N = 100
    p = 20
    patterns = np.random.random_integers(0, 1, size=(p, N))
    n = Network(N, p)
    n.storePatterns(patterns)
    a = 1
