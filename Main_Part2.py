from PatternLibrary import *
from Network import *
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime


startTime = datetime.now()      # for timing script
graphics = False        # for checking visually

iterationsForEachPattern = 5
qSequence = np.linspace(0.0, 1.0, 21)
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

    for i in range(iterationsForEachPattern):

        # distort patterns
        distPatterns = copy.deepcopy(patterns)
        for i, pat in enumerate(distPatterns):
            for j, entry in enumerate(distPatterns[i]):
                r = np.random.uniform(0, 1)
                if r < q:
                    distPatterns[i][j] = - entry

        net = Network(N, p)
        net.storePatterns(patterns, calcW=False)

        for iCorrect, thisInput in enumerate(distPatterns):
            stableState = False
            net.inputPattern(thisInput)
            currentState = []

            if graphics:
                ax.imshow(thisInput.reshape((16, 10)), cmap=plt.cm.binary, interpolation='nearest')
                animation.canvas.draw()

            while stableState == False:
                change = net.updateHopfieldAsynchronous()
                currentState = np.asarray(net.getCurrentNetworkState())

                if graphics:
                    ax.imshow(currentState.reshape((16, 10)), cmap=plt.cm.binary, interpolation='nearest')
                    animation.canvas.draw()

                if change == False:
                    stableState = True

            if np.array_equal(patterns[iCorrect], currentState):
                probRightResultSequence[iQ] += 1

    print 'q = ', q, ' ---> right: ', (1.0/(iterationsForEachPattern*p))*probRightResultSequence[iQ]

    if graphics:
        plt.close("all")

probRightResultSequence = (1.0 / (iterationsForEachPattern * p)) * probRightResultSequence


plt.plot(qSequence, probRightResultSequence, lw=3.0)
plt.title('Deterministic Hopfield Pattern Recognition\nPerformance vs. Level of Pattern Distortion')
plt.ylabel('Relative Frequency of Right Pattern Recognition')
plt.xlabel("Probability of 'Flipping' for Each Bit")
plt.show()



print 'It took ', datetime.now() - startTime, 'to complete this script.'