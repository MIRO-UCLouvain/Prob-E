import numpy as np
from Probabilistic_Evaluation.utils import linearInterpolator
from Probabilistic_Evaluation.utils import timed, Timer

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
        if dosemap.shape != mask.shape:
            raise ValueError("Dosemap and mask must have the same shape.")
        if len(dosemap.shape) != len(spacing):
            raise ValueError("Spacing must have the same number of dimensions as dosemap.")
        for dim in spacing:
            if dim <= 0:
                raise ValueError("All spacing values must be positive.")
        if max_DVH <= 0.0:
            raise ValueError("max_DVH must be a non-negative value.")
        self._dosemap = dosemap
        self._mask = mask
        self._bin_dose = None  # Store bin doses for Dx and Vx calculations, makes Dx calculations easier
        self._dvh = None
        self._maxDVH = max_DVH
        self._DMin = None
        self._DMax = None
        self._DMean = None
        self._spacing = spacing
        self.computeDVH_fast()

    @property
    def dosemap(self) -> np.ndarray:
        return self._dosemap

    @dosemap.setter
    def dosemap(self, newDosemap):
        self._dosemap = newDosemap
        self.computeDVH_fast()

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def maxDVH(self) -> float:
        return self._maxDVH

    @property
    def bin_dose(self) -> np.ndarray:
        return self._bin_dose

    @property
    def dvh(self) -> np.ndarray:
        return self._dvh

    @property
    def DMin(self) -> float:
        if self._DMin is None:
            dose_values = np.where(self.mask>0, self.dosemap, 0)
            self._DMin = np.min(dose_values)
        return self._DMin

    @property
    def DMax(self) -> float:
        if self._DMax is None:
            dose_values = np.where(self.mask>0, self.dosemap, 0)
            self._DMax = np.max(dose_values)
        return self._DMax
    
    @property
    def DMean(self) -> float:
        if self._DMean is None:
            dose_values = np.where(self.mask>0, self.dosemap, 0)
            self._DMean = np.mean(dose_values*self.mask)
        return self._DMean

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

    @timed
    def computeDVH(self):
        """
        Compute Dose-Volume Histogram (DVH) for a given mask.
        """
        n_bins = 4096
        if isinstance(self.mask, np.ndarray):
            #mask = self.mask.astype(bool)
            #dose_mask = self.dosemap[mask]
            #implemented non binary logic to allow for partial volume effects, scale doses according to volumes for easy calculation
            dose_mask = np.where(self.mask>0, self.dosemap, 0)
            dose_mask_scaled = dose_mask*self.mask
            bin_size = self.maxDVH / n_bins
            bin_edges = np.arange(0, self.maxDVH + 0.5 * bin_size, bin_size)  # np.arange is exclusive right limit
            bin_edges[-1] += dose_mask_scaled.max()  # Ensure the max dose is included in the last bin
            hist, _ = np.histogram(dose_mask_scaled, bins=bin_edges)
            hist = np.flip(hist, 0)  # Flip to get descending order
            dvh = np.cumsum(hist)  # Cumulative sum
            dvh = np.flip(dvh, 0)  # Flip back to ascending order
            dvh = dvh / np.sum(hist) * 100.0  # Normalize to percentage
            self._bin_dose = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # dose at midpoints of bins
            self._dvh = dvh

            # Compute DMin, DMax, DMean before deleting dosemap to free memory
            DMax = self.DMax
            DMin = self.DMin
            DMean = self.DMean

            del self._dosemap
            self._dosemap = None  # free memory
    
    @timed
    def computeDVH_fast(self):
        """
        Compute Dose-Volume Histogram (DVH) using a faster CPU-only path.

        The output is kept compatible with computeDVH(): same binning logic,
        same DVH normalization, and same cached summary statistics.
        """
        n_bins = 4096
        if isinstance(self.mask, np.ndarray):
            # mask = self.mask.astype(bool)
            # dose_mask = self.dosemap[mask]
            dose_mask = np.where(self.mask>0, self.dosemap, 0)
            
            bin_size = 101 / n_bins
            bin_edges = np.arange(0, 101 + 0.5 * bin_size, bin_size)
            bin_edges[-1] += dose_mask.max()  # Ensure the max dose is included in the last bin

            bin_idx = np.floor(dose_mask / bin_size).astype(np.int32)
            np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
            hist = np.bincount(bin_idx.flatten(), minlength=n_bins, weights=self.mask.flatten())

            dvh = np.cumsum(hist[::-1])[::-1]
            dvh = dvh / np.sum(hist) * 100.0

            self._bin_dose = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            self._dvh = dvh
            DMax = self.DMax
            DMin = self.DMin
            DMean = self.DMean

            del self._dosemap
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
        target = x * 100.0

        # DVH is decreasing with dose → reverse for interpolation
        dvh_rev = self.dvh[::-1]
        dose_rev = self.bin_dose[::-1]
        Dx = linearInterpolator(target, dvh_rev, dose_rev)
        if Dx>self.DMax:
            Dx = self.DMax
        return Dx

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
        return linearInterpolator(x, self.bin_dose, self.dvh)

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

        total_volume_cc = np.sum(self.mask.astype(bool)) * np.prod(self.spacing) / 1000.0  # convert mm^3 to cc
        if x > total_volume_cc:
            raise ValueError("Absolute volume exceeds total volume of the structure.")

        target_percentage = (x / total_volume_cc) * 100.0

        dvh_rev = self.dvh[::-1]
        dose_rev = self.bin_dose[::-1]
        Dcc = linearInterpolator(target_percentage, dvh_rev, dose_rev)
        if Dcc>self.DMax:
            Dcc = self.DMax
        return Dcc

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
        volume_percentage = linearInterpolator(x, self.bin_dose, self.dvh)

        total_volume_cc = (np.sum(self.mask.astype(bool)) * np.prod(self.spacing) / 1000.0)

        return (volume_percentage / 100.0) * total_volume_cc
