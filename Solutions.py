from Network import *
from PatternLibrary import *
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime


class Solution:
    def solvePartOne(self):
        # plot.ion()
        Ns = [100, 200]
        ps = [10, 20, 30, 40, 50, 75, 100, 150, 200]
        # fig, ax = plot.subplots()
        hs = []
        for Ncounter in range(len(Ns)):
            # line, = plot.plot([])
            # hs.append(line)
            for pcounter in range(len(ps)):
                N = Ns[Ncounter]
                p = ps[pcounter]
                if p > N:
                    break
                iterations = 10
                np.random.seed(seed=int(time.time() % 100))
                patterns = np.random.random_integers(0, 1, size=(p, N))
                patterns[patterns == 0] = -1
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
                while iterations != 0:
                    n.runError(patterns)
                    iterations -= 1
                print("N: " + str(N) + " p: " + str(p) + " : " + str(np.mean(n.errors)))
                # hs[-1].set_xdata([np.append(hs[-1].get_xdata(), p * 1.0 / N)])
                # hs[-1].set_ydata([np.append(hs[-1].get_ydata(), np.mean(n.errors))])
                # #hs[-1].set_data([np.append(hs[-1].get_xdata(), p * 1.0 / N)],[np.append(hs[-1].get_ydata(), np.mean(n.errors))])
                # plot.draw()
                # plot.show()
                # fig.canvas.draw()
                # fig.canvas.flush_events()

    def solvePartTwo(self):
        startTime = datetime.now()  # for timing script
        graphics = False  # for checking visually

        iterationsForEachPattern = 20
        maxIterations=50
        qSequence = np.linspace(0.0, 1.0, 41)
        probRightResultSequence = np.zeros(len(qSequence))

        lib = PatternLibrary()
        lib.saveZeroToFour()
        patterns = np.asarray(lib.savedPatterns)
        N = len(patterns[0])
        p = len(patterns)

        for iQ, q in enumerate(qSequence):

            # set up animation
            if graphics:
                animation = plt.figure()
                title = 'Hopfield Net Asynchronous Updating:  q = ' + str(q)
                plt.title(title)
                ax = animation.gca()
                animation.show()

            net = Network(N, p)
            net.storePatterns(patterns)

            for i, thisInput in enumerate(patterns):

                if graphics:
                    ax.imshow(thisInput.reshape((16, 10)), cmap=plt.cm.binary, interpolation='nearest')
                    animation.canvas.draw()

                for k in range(iterationsForEachPattern):
                    workPattern = net.distortPattern(q, thisInput)
                    for j in range(maxIterations):
                        status = net.runDigits(workPattern, thisInput)
                        currentState = net.getCurrentNetworkState()
                        workPattern = currentState
                        if graphics:
                            ax.imshow(currentState.reshape((16, 10)), cmap=plt.cm.binary, interpolation='nearest')
                            animation.canvas.draw()

                        if status == 1 or status == 2:
                            probRightResultSequence[iQ] += 1
                            break
                        if status == 3:
                            break

            print 'q = ', q, ' ---> right: ', 1.0 * probRightResultSequence[iQ] / (iterationsForEachPattern*p)

            if graphics:
                plt.close("all")

        probRightResultSequence = (1.0 / (iterationsForEachPattern * p)) * probRightResultSequence

        plt.plot(qSequence, probRightResultSequence, lw=3.0)
        plt.title('Deterministic Hopfield Pattern Recognition\nPerformance vs. Level of Pattern Distortion')
        plt.ylabel('Relative Frequency of Right Pattern Recognition')
        plt.xlabel("Probability of 'Flipping' for Each Bit")
        print 'It took ', datetime.now() - startTime, 'to complete this script.'
        plt.show()

    def solvePartThree(self):
        pass

    def solvePartFour(self):
        pass

    def __init__(self, part):
        self.sol[part](self)

    sol = {1: solvePartOne,
           2: solvePartTwo,
           3: solvePartThree,
           4: solvePartFour
           }


if __name__ == '__main__':
    Solution(2)
