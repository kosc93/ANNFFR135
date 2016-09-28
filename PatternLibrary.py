from numpy import array
#from matplotlib import pylab as plt
import time
import numpy as np


# Display all numbers with:
# lib = PatternLibrary()
# lib.saveZeroToFour()
# lib.dispAllPatterns()


class PatternLibrary:
    def __init__(self):
        self.savedPatterns = []

    def toPattern(self, letter):
        pat = letter.replace('\n', '')
        pat = pat.replace(' ', '')
        return array([+1 if c == 'X' else -1 for c in pat])

    # def dispAllSavedPatterns(self, title='Displaying All Saved Patterns'):
    #     animation = plt.figure()
    #     plt.title(title)
    #     ax = animation.gca()
    #     animation.show()
    #     for savePat in self.savedPatterns:
    #         ax.imshow(savePat.reshape((16, 10)), cmap=plt.cm.binary, interpolation='nearest')
    #         animation.canvas.draw()
    #         time.sleep(1)
    #
    # def displayPatterns(self, patterns, title='Displaying Patterns', sleep=0.2):
    #     animation = plt.figure()
    #     plt.title(title)
    #     ax = animation.gca()
    #     animation.show()
    #     for pattern in patterns:
    #         ax.imshow(pattern.reshape((16, 10)), cmap=plt.cm.binary, interpolation='nearest')
    #         animation.canvas.draw()
    #         time.sleep(sleep)

    def saveZeroToFour(self):
        zero = """
        ----------
        ---XXXX---
        --XXXXXX--
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        -XXX--XXX-
        --XXXXXX--
        ---XXXX---
        ----------
        """

        one = """
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        ---XXXX---
        """

        two = """
        XXXXXXXX--
        XXXXXXXX--
        -----XXX--
        -----XXX--
        -----XXX--
        -----XXX--
        -----XXX--
        XXXXXXXX--
        XXXXXXXX--
        XXX-------
        XXX-------
        XXX-------
        XXX-------
        XXX-------
        XXXXXXXX--
        XXXXXXXX--
        """

        three = """
        --XXXXXX--
        --XXXXXXX-
        ------XXX-
        ------XXX-
        ------XXX-
        ------XXX-
        ------XXX-
        ----XXXX--
        ----XXXX--
        ------XXX-
        ------XXX-
        ------XXX-
        ------XXX-
        ------XXX-
        --XXXXXXX-
        --XXXXXX--
        """

        four = """
        -XX----XX-
        -XX----XX-
        -XX----XX-
        -XX----XX-
        -XX----XX-
        -XX----XX-
        -XX----XX-
        -XXXXXXXX-
        -XXXXXXXX-
        -------XX-
        -------XX-
        -------XX-
        -------XX-
        -------XX-
        -------XX-
        -------XX-
        """

        numberStrings = [zero, one, two, three, four]
        allNumbers = [self.toPattern(x) for x in numberStrings]
        for number in allNumbers:
            self.savedPatterns.append(number)

    def loadDataSet(self, fileName, normalize=True):
        zeta=[]
        xi=[]
        data=open(fileName,"r")
        for line in data:
            parts=line.split(" ")
            xi.append([float(parts[0]), float(parts[1])])
            zeta.append(int(parts[2]))
        data.close()
        xi=np.array(xi)
        zeta=np.array(zeta)
        if normalize:
            xi[:, 0] = xi[:,0] / np.std(xi[:, 0])
            xi[:, 0] = xi[:, 0] - np.mean(xi[:, 0])
            xi[:,1]=xi[:,1]/np.std(xi[:,1])
            xi[:,1]=xi[:,1]-np.mean(xi[:,1])
        return zeta,xi

