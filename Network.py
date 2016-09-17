from Neuron import *
import random as rand
import numpy as np
import matplotlib.pyplot as plot
import time, math


# todo calculate weight matrix in network class and store it in neurons for more flexibility


class Network:
    neurons = []
    errors = []
    def __init__(self, N, p):
        rand.seed(rand.random())
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
        self.errors.append(self.neurons[rand.randint(0, len(self.neurons)-1)].singleStep(r))


if __name__ == '__main__':
    N = 100
    p = 40
    iterations=1000
    np.random.seed(seed=int(time.time()%100))
    patterns = np.random.random_integers(0, 1, size=(p, N))
    patterns[patterns==0] = -1
    n = Network(N, p)
    n.storePatterns(patterns)
    h1,=plot.plot([],[])
    # res=[]
    # resw = []
    # for neuron in n.neurons:
    #     res.append(neuron.storedPatterns)
    #     resw.append(neuron.weights)
    # fig = plot.figure()
    # ax=fig.add_subplot(311)
    # ax.matshow(patterns)
    # ax = fig.add_subplot(312)
    # ax.matshow(np.transpose(res))
    # ax = fig.add_subplot(313)
    # ax.matshow(resw)
    # plot.show()
    while iterations!=0:
        n.runHopfield(patterns)
        iterations-=1
        if iterations%10==0:
            print(np.mean(n.errors))
            h1.set_xdata(np.append(h1.get_xdata(), iterations))
            h1.set_ydata(np.append(h1.get_ydata(), np.mean(n.errors)))
            plot.draw()
    pass
