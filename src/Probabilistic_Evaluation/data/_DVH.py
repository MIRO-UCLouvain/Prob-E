import numpy as np
from Probabilistic_Evaluation.utils import linearInterpolator
from Probabilistic_Evaluation.utils import timed, Timer
from Probabilistic_Evaluation.logging_utils import log_call, logger

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
        logger.debug(f"DVH initialized with dosemap shape {dosemap.shape}, mask shape {mask.shape}, spacing {spacing}.")
        logger.debug(f"Max: {self.DMax}, Min: {self.DMin}, Mean: {self.DMean}.")
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
            self._DMin = np.min(self.dosemap[self.mask>0])
        return self._DMin

    @property
    def DMax(self) -> float:
        if self._DMax is None:
            self._DMax = np.max(self.dosemap[self.mask>0])
        return self._DMax
    
    @property
    def DMean(self) -> float:
        if self._DMean is None:
        
            self._DMean = np.average(self.dosemap[self.mask>0], weights=self.mask[self.mask>0])
            
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
    def computeDVH_fast(self):
        """
        Compute Dose-Volume Histogram (DVH) using a faster CPU-only path.

        The output is kept compatible with computeDVH(): same binning logic,
        same DVH normalization, and same cached summary statistics.
        """
        n_bins = 4096
        if isinstance(self.mask, np.ndarray):
            # select the ROI voxels once: every `mask > 0` is a full pass over the dose grid
            inside = self.mask > 0
            doses = self.dosemap[inside]
            weights = self.mask[inside]
            if doses.size == 0:
                logger.debug(f"DVH mask has no voxel above 0 (mask shape {self.mask.shape}): the DVH of an empty ROI cannot be computed.")
            bin_size = 101 / n_bins
            bin_edges = np.arange(0, 101 + 0.5 * bin_size, bin_size)
            bin_edges[-1] += doses.max()  # Ensure the max dose is included in the last bin

            bin_idx = np.floor(doses / bin_size).astype(np.int32)
            np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
            hist = np.bincount(bin_idx, minlength=n_bins, weights=weights)

            dvh = np.cumsum(hist[::-1])[::-1]
            dvh = dvh / np.sum(hist) * 100.0

            self._bin_dose = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            self._dvh = dvh
            # fill the cached statistics from the same selection (the properties would redo the full-grid pass)
            if self._DMax is None:
                self._DMax = np.max(doses)
            if self._DMin is None:
                self._DMin = np.min(doses)
            if self._DMean is None:
                self._DMean = np.average(doses, weights=weights)
            if self._DMax > 101:
                logger.debug(f"Max dose {self._DMax} exceeds the 101 Gy range of the DVH bins: doses above 101 Gy share the last bin, so dose-volume values above 101 Gy are coarse.")

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

        total_volume_cc = np.sum(self.mask) * np.prod(self.spacing) / 1000.0  # convert mm^3 to cc
        if x > total_volume_cc:
            # the requested volume cannot be reached: the dose received by "at least x cc" is the minimum dose
            logger.warning(
                f"Requested absolute volume {x} cc exceeds the structure volume ({total_volume_cc:.4f} cc); "
                "returning Dmin as the dose to that volume."
            )
            return self.DMin

        target_percentage = (x / total_volume_cc) * 100.0

        dvh_rev = self.dvh[::-1]
        dose_rev = self.bin_dose[::-1]
        Dcc = linearInterpolator(target_percentage, dvh_rev, dose_rev)
        if Dcc>self.DMax:
            Dcc = self.DMax
        logger.debug(f"Dose at which {x} cc of the volume receives at least that dose is {Dcc}. Total volume is {total_volume_cc} cc.")
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

        total_volume_cc = (np.sum(self.mask) * np.prod(self.spacing) / 1000.0)
        logger.debug(f"Absolute volume for dose {x} Gy is {volume_percentage}% of total volume {total_volume_cc} cc.")
        return (volume_percentage / 100.0) * total_volume_cc
