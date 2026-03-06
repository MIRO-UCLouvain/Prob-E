import numpy as np

from ._clinicalGoal import AbstractClinicalGoal

class DCCClinicalGoal(AbstractClinicalGoal):
    """
    A class to represent a Dose-Volume Clinical Goal (DCC) for radiation therapy evaluation.

    Attributes
    ----------
    prescription : float
        The prescribed dose value for the clinical goal.
    lower_is_better : bool (default=True)
        Indicates if lower values are better for this goal.
    mask : np.ndarray
        The mask defining the region of interest for the clinical goal.
    maskName : str
        The name of the mask defining the region of interest for the clinical goal.
    priority : int
        The priority level of the clinical goal, zero if no cummulative evaluation is desired.
    valueList : list
        A list to store list of values computed for the clinical goal on each scenario.
    successList : list
        A list to store success status for each scenario.
    volume : float
        The volume (in cc) associated with the clinical goal.
    """

    def __init__(self, prescription: float, mask: np.ndarray, maskName: str, lower_is_better: bool = True, priority: int = 0, **kwargs):
        super().__init__(prescription, mask, maskName, lower_is_better, priority)
        self._volume = kwargs.get('volume', None)  # Volume in cc
        if self._volume is None:
            raise ValueError("Volume must be provided for DCC ClinicalGoal.")
        if self._volume < 0:
            raise ValueError("Volume must be a positive value.")


    @property
    def volume(self) -> float:
        return self._volume
    @volume.setter
    def volume(self, newVolume: float):
        if newVolume < 0:
            raise ValueError("Volume must be a positive value.")
        self._volume = newVolume

    def __str__(self):
        if self.lower_is_better:
            comparison = "<="
        else:
            comparison = ">="
        return f"D{self.volume:.2f}{comparison}{self.prescription}"

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