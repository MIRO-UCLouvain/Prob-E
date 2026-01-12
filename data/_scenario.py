import numpy as np
from data._DVH import DVH
from matplotlib import pyplot as plt

class Scenario():
    """
    Class to represent a scenario with dose distribution and associated DVHs.
    Attributes:
        dosemap (np.ndarray): 3D array representing the dose distribution.
    Methods:
        (To be implemented)
    """
    def __init__(self, dosemap: np.ndarray):
        self.dosemap = dosemap
        self.masks = None
        self.DVHs = None
        self.names = None
        

    def getAllDVH(self, maxDVH: float = 100.0):
        """
        Compute DVHs for all masks.
        """
        self.DVHs = []
        for mask in self.masks:
            self.DVHs.append(DVH(self.dosemap, mask, maxDVH))


    def plot_DVHs(self):

            cmap = plt.cm.get_cmap('tab20', max(1, len(self.DVHs)))
            plt.figure()
            for i in range(len(self.DVHs)):
                dvh = self.DVHs[i]
                name = self.names[i]
                x = np.linspace(0.0, self._maxDVH, len(dvh))
                color = cmap(i if len(self.DVHs) <= 20 else (i % 20))
                plt.plot(x, dvh, label=name, color=color)
            plt.xlim(0, self.DVHs[0].maxDVH)
            plt.ylim(0, 100)
            plt.xlabel('Dose (Gy)')
            plt.ylabel('Volume (%)')
            plt.title('Dose-Volume Histograms')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        