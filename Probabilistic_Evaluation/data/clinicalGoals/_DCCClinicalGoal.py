import numpy as np

from _clinicalGoal import AbstractClinicalGoal

class DCCClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a Dose-Volume Clinical Goal (DCC) for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed dose value for the clinical goal.
    mask : np.ndarray
        A binary mask defining the region of interest for the clinical goal.
    lower_is_better : bool
        Indicates if lower dose values are better for this goal.
    volume : float
        The volume (in cc) associated with the clinical goal.
    """

    def __init__(self, prescription: float, mask: np.ndarray, lower_is_better: bool = True, **kwargs):
        super().__init__(prescription, mask, lower_is_better)
        self._volume = kwargs.get('volume', None)  # Volume in cc
        if self._volume is None:
            raise ValueError("Volume must be provided for DCC ClinicalGoal.")
        if self._volume <= 0:
            raise ValueError("Volume must be a positive value.")


    @property
    def volume(self) -> float:
        return self._volume
    @volume.setter
    def volume(self, newVolume: float):
        if newVolume <= 0:
            raise ValueError("Volume must be a positive value.")
        self._volume = newVolume

    def compute_value(self, dvh) -> float:
        """
        Compute the dose corresponding to the specified volume from the DVH.
        Parameters
        ----------
        dvh : DVH
            The dose-volume histogram object used for computation.

        Returns
        -------
        float
            The computed dose value corresponding to the specified volume.
        """
        value  = dvh.computeDcc(self.volume)
        return value