import numpy as np


class DVH(object):
    """
    Class to compute Dose-Volume Histogram (DVH) from dose distribution and a mask.
    Attributes
    ----------
    dosemap : np.ndarray
        The 3D dose distribution array.
    mask : np.ndarray
        The binary mask defining the region of interest.
    maxDVH : float
        The maximum dose value for DVH calculation.
    bin_dose : np.ndarray
        The dose values corresponding to the DVH bins.
    dvh : np.ndarray
        The computed DVH values as a percentage of volume.
    DMin : float
        The minimum dose within the masked region.
    DMax : float
        The maximum dose within the masked region.
    DMean : float
        The mean dose within the masked region.
    spacing : tuple (default=(1.0, 1.0, 1.0))
        A tuple representing the voxel spacing in each dimension (x, y, z).

    Methods
    -------
    computeDVH()
        Computes the DVH for the given mask.
    computeDx(x: float) -> float
        Computes the dose at which x% of the volume receives at least that dose.
    computeVx(x: float) -> float
        Computes the volume percentage receiving at least x Gy dose.
    """

    def __init__(self, dosemap: np.ndarray, mask: np.ndarray, max_DVH: float = 100.0, spacing: tuple = (1.0, 1.0, 1.0)):
        self._dosemap = dosemap
        self._mask = mask
        self._bin_dose = None  # Store bin doses for Dx and Vx calculations, makes Dx calculations easier
        self._dvh = None
        self._maxDVH = max_DVH
        dmin = np.min(self.dosemap[self.mask.astype(bool)])
        self._DMin = dmin
        dmax = np.max(self.dosemap[self.mask.astype(bool)])
        self._DMax = dmax
        self.DMax
        dmean = np.mean(self.dosemap[self.mask.astype(bool)])
        self._Dmean = dmean
        self.DMean
        self._spacing = spacing
        self.computeDVH()
    @property
    def dosemap(self) -> np.ndarray:
        return self._dosemap

    @dosemap.setter
    def dosemap(self, newDosemap):
        self._dosemap = newDosemap

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, newMask: np.ndarray):
        self._mask = newMask

    @property
    def maxDVH(self) -> float:
        return self._maxDVH

    @maxDVH.setter
    def maxDVH(self, newMaxDVH: float):
        if newMaxDVH <= 0:
            raise ValueError("maxDVH must be a positive value.")
        self._maxDVH = newMaxDVH

    @property
    def bin_dose(self) -> np.ndarray:
        return self._bin_dose

    @property
    def dvh(self) -> np.ndarray:
        return self._dvh

    @property
    def DMin(self) -> float:
        return self._Dmin

    @DMin.setter
    def DMin(self, newDMin: float):
        if newDMin < 0:
            raise ValueError("DMin must be a non-negative value.")
        self._Dmin = newDMin

    @property
    def DMax(self) -> float:
        return self._Dmax

    @DMax.setter
    def DMax(self, newDMax: float):
        if newDMax < 0:
            raise ValueError("DMax must be a non-negative value.")
        self._Dmax = newDMax

    @property
    def DMean(self) -> float:
        return self._Dmean

    @DMean.setter
    def DMean(self, newDMean: float):
        if newDMean < 0:
            raise ValueError("DMean must be a non-negative value.")
        self._Dmean = newDMean

    @property
    def spacing(self) -> tuple:
        return self._spacing

    @spacing.setter
    def spacing(self, newSpacing: tuple):
        if len(newSpacing) != 3:
            raise ValueError("Spacing must be a list of three float values.")
        if any(s <= 0 for s in newSpacing):
            raise ValueError("All spacing values must be positive.")
        self._spacing = newSpacing

    def computeDVH(self):
        """
        Compute Dose-Volume Histogram (DVH) for a given mask.

        """
        n_bins = 4096
        mask = self.mask.astype(bool)
        dose_mask = self.dosemap[mask]
        bin_size = self.maxDVH / n_bins
        bin_edges = np.arange(0, self.maxDVH + 0.5 * bin_size, bin_size)  # np.arange is exclusive right limit
        bin_edges[-1] += dose_mask.max()  # Ensure the max dose is included in the last bin
        hist, _ = np.histogram(dose_mask, bins=bin_edges)
        hist = np.flip(hist, 0)  # Flip to get descending order
        dvh = np.cumsum(hist)  # Cumulative sum
        dvh = np.flip(dvh, 0)  # Flip back to ascending order
        dvh = dvh / np.sum(hist) * 100.0  # Normalize to percentage
        self._bin_dose = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # dose at midpoints of bins
        self._dvh = dvh
        self._dosemap = None  # free memory

    def computeDx(self, x: float) -> float:
        """
        Compute the dose at which x% of the volume receives at least that dose.
        Parameters
        ----------
        x : float
            The volume percentage (0-1).

        Returns
        -------
        float
            The computed dose value corresponding to the specified volume percentage.
        """
        if x < 0:
            raise ValueError("Volume percentage must be a non-negative value.")
        if x > 1:
            raise ValueError("Volume percentage must be given in percentage (0-1).")
        diff_array = np.abs(self.dvh - x*100.0)
        index = np.argmin(diff_array)
        return self.bin_dose[index]

    def computeVx(self, x: float) -> float:
        """
        Compute the volume percentage receiving at least x Gy dose.
        Parameters
        ----------
        x : float
            The dose value in Gy.

        Returns
        -------
        float
            The computed volume percentage corresponding to the specified dose.
        """
        if x < 0:
            raise ValueError("Dose must be a non-negative value.")
        diff_array = np.abs(self.bin_dose - x)
        index = np.argmin(diff_array)
        return self.dvh[index]

    def computeDcc(self, x: float) -> float:
        """
        Compute the dose at which x cc of the volume receives at least that dose.
        Parameters
        ----------
        x : float
            The absolute volume in cc.

        Returns
        -------
        float
            The computed dose value corresponding to the specified absolute volume.
        """
        if x < 0:
            raise ValueError("Absolute volume must be a non-negative value.")
        total_volume_cc = np.sum(self.mask.astype(bool))*np.prod(self.spacing)/1000.0  # convert mm^3 to cc
        volume_percentage = (x / total_volume_cc) * 100.0
        diff_array = np.abs(self.dvh - volume_percentage)
        index = np.argmin(diff_array)
        return self.bin_dose[index]

    def computeVcc(self, x: float) -> float:
        """
        Compute the absolute volume in cc receiving at least x Gy dose.
        Parameters
        ----------
        x : float
            The dose value in Gy.

        Returns
        -------
        float
            The computed absolute volume in cc corresponding to the specified dose.
        """
        if x < 0:
            raise ValueError("Dose must be a non-negative value.")
        diff_array = np.abs(self.bin_dose - x)
        index = np.argmin(diff_array)
        volume_percentage = self.dvh[index]
        total_volume_cc = np.sum(self.mask.astype(bool))*np.prod(self.spacing)/1000.0  # convert mm^3 to cc
        absolute_volume_cc = (volume_percentage / 100.0) * total_volume_cc
        return absolute_volume_cc
