from Neuron import *
import random as rand

class Network:
    neurons = []

    def __init__(self,N,p):
        for i in range(N):
            n = Neuron(i,N)
            self.neurons.append(n)
        for i in range(len(self.neurons)):
            self.neurons[i].connectHopfield(self.neurons)

    def storePatterns(self, patterns):
        for i in range(len(patterns)):
            for j in range(len(self.neurons)):
                self.neurons[j].storePattern(patterns[i])
        for i in range(len(self.neurons)):
            self.neurons[i].calcWeights()

    def run(self):
        i = rand.randint(0,len(self.neurons))
        self.neurons[i].singleStep()

if __name__ == '__main__':
    n= Network(10,1)
    a = 1
