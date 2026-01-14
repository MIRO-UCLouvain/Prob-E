import numpy as np
from abc import ABC, abstractmethod

from Probabilistic_Evaluation.data.uncertaintyModel._abstractUncertaintyModel import AbstractUncertaintyModel


class AbstractsamplingMethod(ABC):
    """
    Abstract base class for sampling methods.

    Attributes
    ----------
    UncertaintyModel : AbstractUncertaintyModel
        The uncertainty model used for sampling.

    Methods
    -------
    MCsampling()
        Abstract method for Monte Carlo sampling.
    analyticalSampling()
        Abstract method for analytical sampling.
    """

    def __init__(self, uncertaintyModel: AbstractUncertaintyModel):
        super().__init__()
        self.UncertaintyModel = uncertaintyModel

    @abstractmethod
    def MCsampling(self):
        pass

    @abstractmethod
    def analyticalSampling(self):
        pass

