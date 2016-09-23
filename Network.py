from Neuron import *
import random as rand
import numpy as np
import matplotlib.pyplot as plot
import time, math


# todo calculate weight matrix in network class and store it in neurons for more flexibility


class Network:
    def __init__(self, N, p):
        self.neurons=[]
        self.errors=[]
        rand.seed(rand.random())
        for i in range(N):
            neuron=Neuron(i, N)
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
            self.errors.append(self.neurons[rand.randint(0, len(self.neurons)-1)].singleStep(r))

    def inputPattern(self, input):
        for i in range(len(self.neurons)):
            self.neurons[i].state = input[i]

    def getCurrentNetworkState(self):
        netState = []
        for i in range(len(self.neurons)):
            netState.append(self.neurons[i].state)
        return netState

    def updateHopfieldAsynchronous(self):
        stateChange = False
        for j in np.random.permutation(len(self.neurons)):      # update all neurons once, but in random order
            change = self.neurons[j].asynchronousUpdate()
            if change == True:
                stateChange = True
        return stateChange



if __name__ == '__main__':
    plot.ion()
    Ns=[100,200]
    ps=[10, 20,30,40,50,75,100,150,200]
    #fig, ax = plot.subplots()
    hs=[]
    for Ncounter  in range(len(Ns)):
        line, = plot.plot([])
        hs.append(line)
        for pcounter in range(len(ps)):
            N = Ns[Ncounter]
            p = ps[pcounter]
            if p>N:
                break
            iterations=10
            np.random.seed(seed=int(time.time()%100))
            patterns = np.random.random_integers(0, 1, size=(p, N))
            patterns[patterns==0] = -1
            n = Network(N, p)
            n.storePatterns(patterns)

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
            print("N: "+ str(N) + " p: "+ str(p) + " : " + str(np.mean(n.errors)))
            hs[-1].set_xdata([np.append(hs[-1].get_xdata(), p * 1.0 / N)])
            hs[-1].set_ydata([np.append(hs[-1].get_ydata(), np.mean(n.errors))])
            #hs[-1].set_data([np.append(hs[-1].get_xdata(), p * 1.0 / N)],[np.append(hs[-1].get_ydata(), np.mean(n.errors))])
            plot.draw()
            plot.show()
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            pass
