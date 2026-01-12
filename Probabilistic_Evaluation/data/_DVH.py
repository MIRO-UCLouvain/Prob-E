import numpy as np
import matplotlib.pyplot as plt


class DHV:
    """
    Class to compute Dose-Volume Histograms (DVHs) from dose distributions and masks.

    Attributes:
    ----------
    dosemap : np.ndarray
        3D array representing the dose distribution.
    names : list of str
        List of names for each mask/structure.
    masks : list of np.ndarray
        List of boolean masks for regions of interest.
    dvhs : list of np.ndarray
        List of computed DVHs for each mask.
    dmax : list of float
        List of maximum doses for each mask.
    dmin : list of float
        List of minimum doses for each mask.
    """

    def __innit__(self, dosemap: np.ndarray, maskDict:dict, max_DVH: float = 100.0):
        self._dosemap = dosemap
        self._names = list(maskDict.keys())
        self._masks = [maskDict[name].astype(bool) for name in self._names]
        self._bin_doses = None  # Store bin doses for DVH calculations, makes Dx calculations easier
        self._dvhs = None

        self._Dmean = None
        self._Dmax = None
        self._Dmin = None
        self._D98 = None
        self._D95 = None
        self._D50 = None
        self._D5 = None
        self._D2 = None

        self._maxDVH = max_DVH

    @property
    def dosemap(self) -> np.ndarray:
        return self._dosemap

    @property
    def names(self) -> list[str]:
        return self._names

    @property
    def mask(self) -> np.ndarray:
        return self._masks

    @property
    def dvhs(self) -> list[np.ndarray]:
        if self._dvhs is None:
            self._dvhs = self.getAllDVH()
        return self._dvhs

    @property
    def dmax(self) -> list[float]:
        if self._dmax is None:
            self.getAllDVH()
        return self._dmax

    @property
    def dmin(self) -> list[float]:
        if self._dmin is None:
            self.getAllDVH()
        return self._dmin

    @property
    def bin_doses(self) -> list[np.ndarray]:
        if self._bin_doses is None:
            self.getAllDVH()
        return self._bin_doses

    @property
    def Dmean(self) -> list[float]:
        if self._Dmean is None:
            self.getAllDVH()
        return self._Dmean

    @property
    def D98(self) -> list[float]:
        if self._D98 is None:
            self.getAllDVH()
        return self._D98

    @property
    def D95(self) -> list[float]:
        if self._D95 is None:
            self.getAllDVH()
        return self._D95

    @property
    def D50(self) -> list[float]:
        if self._D50 is None:
            self.getAllDVH()
        return self._D50

    @property
    def D5(self) -> list[float]:
        if self._D5 is None:
            self.getAllDVH()
        return self._D5

    @property
    def D2(self) -> list[float]:
        if self._D2 is None:
            self.getAllDVH()
        return self._D2

    def computeDVH(self, mask, max_DVH=100.0) -> np.ndarray:
        """
        Compute Dose-Volume Histogram (DVH) for a given mask.
        Parameters:
            mask (np.ndarray): A boolean mask array where True indicates the region of interest.
        Returns:
            dvh (1D np.array): The computed DVH as a percentage array. in the range [0, max_DVH]Gy .
        """
        n_bins = 4096
        mask = mask.astype(bool)
        dose_mask = self._dosemap[mask]
        bin_size = max_DVH / n_bins
        bin_edges = np.arange(0, max_DVH + 0.5 * bin_size, bin_size)  # np.arange is exclusive right limit
        bin_edges[-1] += dose_mask.max()  # Ensure the max dose is included in the last bin
        bin_dose = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Midpoints of bins
        hist, _ = np.histogram(dose_mask, bins=bin_edges)
        hist = np.flip(hist, 0)  # Flip to get descending order
        dvh = np.cumsum(hist)  # Cumulative sum
        dvh = np.flip(dvh, 0)  # Flip back to ascending order
        dvh = dvh / np.sum(hist) * 100.0  # Normalize to percentage

        dmean = np.mean(dose_mask)
        dmax = dose_mask.max()
        dmin = dose_mask.min()
        D98 = self.computeDx(dvh, bin_dose, 98)
        D95 = self.computeDx(dvh, bin_dose, 95)
        D50 = self.computeDx(dvh, bin_dose, 50)
        D5 = self.computeDx(dvh, bin_dose, 5)
        D2 = self.computeDx(dvh, bin_dose, 2)
        return dvh, bin_dose, dmean, dmax, dmin, D98, D95, D50, D5, D2

    def getAllDVH(self):
        """
        Compute DVHs for all masks.
        """
        dvhs = []
        dmins = []
        dmaxs = []
        bin_doses = []
        dmeans = []
        D98s = []
        D95s = []
        D50s = []
        D5s = []
        D2s = []
        for mask in self._mask:
            dvh, bin_dose, dmean, dmax, dmin, D98, D95, D50, D5, D2 = self.computeDVH(mask, self._maxDVH)
            dvhs.append(dvh)
            dmaxs.append(dmax)
            dmins.append(dmin)
            bin_doses.append(bin_dose)
            D98s.append(D98)
            D95s.append(D95)
            D50s.append(D50)
            D5s.append(D5)
            D2s.append(D2)
        self._Dmax = dmaxs
        self._Dmin = dmins
        self._dvhs = dvhs
        self._bin_doses = bin_doses
        self._Dmean = dmeans
        self._D98 = D98s
        self._D95 = D95s
        self._D50 = D50s
        self._D5 = D5s
        self._D2 = D2s

    def computeDx(self, dvh: np.ndarray, bin_doses: np.ndarray, x: float) -> float:
        diff_array = np.abs(dvh - x)
        index = np.argmin(diff_array)
        return bin_doses[index]

    def print_DVHs(self):
        self.getAllDVH()  # Ensure DVHs are computed

        cmap = plt.cm.get_cmap('tab20', max(1, len(self.dvhs)))
        plt.figure()
        for i in range(len(self._dvhs)):
            dvh = self._dvhs[i]
            name = self._names[i]
            x = np.linspace(0.0, self._maxDVH, len(dvh))
            color = cmap(i if len(self._dvhs) <= 20 else (i % 20))
            plt.plot(x, dvh, label=name, color=color)
        plt.xlim(0, self._maxDVH)
        plt.ylim(0, 100)
        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume (%)')
        plt.title('Dose-Volume Histograms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
