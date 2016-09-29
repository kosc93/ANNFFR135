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
                iterations = 100
                np.random.seed(seed=int(time.time() % 100))
                patterns = np.random.random_integers(0, 1, size=(p, N))
                patterns[patterns == 0] = -1
                n = Network(N)
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

            set up animation
            if graphics:
                animation = plt.figure()
                title = 'Hopfield Net Asynchronous Updating:  q = ' + str(q)
                plt.title(title)
                ax = animation.gca()
                animation.show()

            net = Network(N)
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
        startTime = datetime.now()  # for timing script
        plotAtEnd = False
        iterationPerPattern=1
        maxIterations=500
        beta=2
        transientLength = 200
        p50 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
        p100 = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
        p250 = np.array([0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
        p500 = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50])
        NSequence = np.array([50, 100, 250, 500])
        pSequence = np.array([p50, 2*p100, 5*p250, 10*p500])

        alphaSequence50 = 1.0 * pSequence[0] / NSequence[0]
        alphaSequence100 = 1.0 * pSequence[1] / NSequence[1]
        alphaSequence250 = 1.0 * pSequence[2] / NSequence[2]
        alphaSequence500 = 1.0 * pSequence[3] / NSequence[3]
        mForPlot = []
        for thisPIndex, thisN in enumerate(NSequence):
            mSequence = []
            print '\nthisN =', thisN
            thisP = pSequence[thisPIndex]
            for pat in thisP:
                print "patterns:", pat,
                mMean=[]
                print '---> iteration:',
                for pIteration in range(iterationPerPattern):
                    print pIteration + 1,
                    m = []
                    patterns = np.random.random_integers(0, 1, size=(int(pat), thisN))
                    patterns[patterns == 0] = -1
                    n = Network(thisN, beta)
                    n.storePatterns(patterns)
                    workPattern=patterns[0]
                    for nIteration in range(maxIterations):
                        n.runStochastic(workPattern)
                        workPattern = n.getCurrentNetworkState()
                        if nIteration >= transientLength:
                            m.append(n.calcMMu(1))
                    mMean.append(np.mean(m))
                mSequence.append(np.mean(mMean))
                print 'alpha:', pat*1.0/thisN, ' m:', np.mean(mMean),
                print datetime.now() - startTime
            mForPlot.append(mSequence)

        print 'alphaSequence50:', pprint.pprint(alphaSequence50)
        print 'alphaSequence100:', pprint.pprint(alphaSequence100)
        print 'alphaSequence250:', pprint.pprint(alphaSequence250)
        print 'alphaSequence500:', pprint.pprint(alphaSequence500)
        print 'mForPlot[0]', mForPlot[0]
        print 'mForPlot[1]', mForPlot[1]
        print 'mForPlot[2]', mForPlot[2]
        print 'mForPlot[3]', mForPlot[3]

        if plotAtEnd:
            N50, = plt.plot(alphaSequence50, mForPlot[0], label='N = 50')
            N100, = plt.plot(alphaSequence100, mForPlot[1], label='N = 100')
            N250, = plt.plot(alphaSequence250, mForPlot[2], label='N = 250')
            N500, = plt.plot(alphaSequence500, mForPlot[3], label='N = 500')
            plt.legend(handles=[N50, N100, N250, N500], loc=1) #upper right
            plt.xlabel(r'$\alpha$', fontsize=18)
            plt.ylabel(r'$m$', fontsize=18)
            plt.title(r'Order parameter $m$ as a function of $\alpha = p/N$', fontsize=14)
            plt.show()

        print 'It took ', datetime.now() - startTime, 'to complete this script.'


    def solvePartFour(self):
        startTime = datetime.now()  # for timing script
        graphics=False
        plotAtEnd=False
        iterations=100000
        numTrainings=100
        numHiddenSequence = [0, 2, 4, 8, 16, 32]
        meanTrainErrorSequence = []
        meanValidErrorSequence = []
        beta=0.5
        eta=0.01
        Input=range(2)
        lib=PatternLibrary()
        zetas,xis=lib.loadDataSet("trainingSet.txt")
        validZetas, validXis=lib.loadDataSet("validationSet.txt")
        xi1, xi2 = xis[:,0], xis[:,1]
        outputlines, hiddenlines = [], []

        for numHidden in numHiddenSequence:
            thisPartStartTime = datetime.now()
            Hidden = range(2, 2 + numHidden)
            Output = [2 + numHidden]
            N = len(Input) + len(Output) + len(Hidden)
            trainErrors, validErrors = [], []
            print 'Trainings completed (out of 100):',

            for training in range(numTrainings):
                minTrainError, minValidError = 1, []

                if graphics:
                    fig = plt.figure()
                    t = np.linspace(-3, 3, num=10)
                    if len(Hidden)==0:
                        for i in range(len(Output)):
                            outputlines.append(plt.plot(np.arange(10),'b')[0])
                    for i in range(len(Hidden)):
                        hiddenlines.append(plt.plot(np.arange(10),'--c')[0])
                    plt.xlim([-3, 3])
                    plt.ylim([-3, 3])
                    plt.plot(xi1[zetas == 1], xi2[zetas == 1], 'xr')
                    plt.plot(xi1[zetas == -1], xi2[zetas == -1], 'xb')
                    fig.show()

                n = Network(N, beta, Input, Output, Hidden, eta)

                for iter in range(iterations):
                    index = int(np.random.randint(0,len(xis)))  # choose random pattern from training set
                    xi, zeta = xis[index], zetas[index]
                    n.trainFF(xi,zeta)  # train network on that pattern

                    if iter%1000==0 or iter>99900:
                        tError = n.calcError(zetas, xis)  # error for training set
                        if tError < minTrainError:
                            minTrainError = tError;
                            minValidError = n.calcError(validZetas, validXis)  # error for validation set

                            if graphics:
                                for i,ih in enumerate(hiddenlines):
                                    ih.set_ydata((n.hiddens[i].bias-n.hiddens[i].weights[0]*t)/n.hiddens[i].weights[1])
                                    ih.set_xdata(t)
                            if graphics:
                                plt.plot((n.outputs[0].bias - n.outputs[0].weights[0] * t) / n.outputs[0].weights[1], t, '--m')
                                print 'error: ', error, ' w0: ', n.outputs[0].weights[0] , ' w1: ', n.outputs[0].weights[1], ' bias: ', n.outputs[0].bias

                        if graphics:
                            for i,io in enumerate(outputlines):
                                io.set_ydata((n.outputs[i].bias-n.outputs[i].weights[0]*t)/n.outputs[i].weights[1])
                                io.set_xdata(t)
                            fig.canvas.draw()

                trainErrors.append(minTrainError)
                validErrors.append(minValidError)
                print training + 1

            print '\nHidden Neurons =', numHidden
            print 'Training    --->  Mean Error:', "{:.3f}".format(np.mean(trainErrors)), '  Standard Deviation:', "{:.3f}".format(np.std(trainErrors))
            print 'Validation  --->  Mean Error:', "{:.3f}".format(np.mean(validErrors)), '  Standard Deviation:', "{:.3f}".format(np.std(validErrors))
            print 'This part took: ', datetime.now() - thisPartStartTime

            meanTrainErrorSequence.append(np.mean(trainErrors))
            meanValidErrorSequence.append(np.mean(validErrors))

        if plotAtEnd:
            train, = plt.plot(numHiddenSequence, meanTrainErrorSequence, label='Training')
            valid, = plt.plot(numHiddenSequence, meanValidErrorSequence, label='Validation')
            plt.legend(handles=[train,valid], loc=1)  # upper right
            plt.xlabel('number of neurons in hidden layer', fontsize=16)
            plt.ylabel('classification error', fontsize=16)
            plt.title('Classification Error for Backpropagation', fontsize=20)
            plt.show()

        print '\nTotal time passed: ', datetime.now() - startTime
            
    def __init__(self, part):
        np.random.seed()
        self.sol[part](self)

    sol = {1: solvePartOne,
           2: solvePartTwo,
           3: solvePartThree,
           4: solvePartFour
           }


if __name__ == '__main__':
    Solution(4)
