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
    def __init__(self, displacement: np.ndarray ,dosemap: np.ndarray, masks: dict[str, np.ndarray]):
        self.displacement = displacement
        self.dosemap = dosemap
        self.masks = masks
     
        self.DVHs = None
        

    def getAllDVH(self, maxDVH: float = 100.0):
        """
        Compute DVHs for all masks.
        """
        self.DVHs = {}
        for name, mask in self.masks.items():
            self.DVHs[name] = DVH(self.dosemap, mask, maxDVH)


    def plot_DVHs(self):

            cmap = plt.cm.get_cmap('tab20', max(1, len(self.DVHs)))
            plt.figure()
            for i, (name, dvh_obj) in enumerate(self.DVHs.items()):
                dvh = dvh_obj.dvh
                x = np.linspace(0.0, dvh_obj.maxDVH, len(dvh))
                color = cmap(i if len(self.DVHs) <= 20 else (i % 20))
                plt.plot(x, dvh, label=name, color=color)
            plt.xlim(0, list(self.DVHs.values())[0].maxDVH)
            plt.ylim(0, 100)
            plt.xlabel('Dose (Gy)')
            plt.ylabel('Volume (%)')
            plt.title('Dose-Volume Histograms')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        