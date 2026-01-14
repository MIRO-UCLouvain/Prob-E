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
    generatedDisplacements : np.ndarray
        Array of generated displacement vectors.
    generatedProbabilities : np.ndarray
        Array of probabilities associated with each displacement vector.

    Methods
    -------
    MCsampling()
        Abstract method for Monte Carlo sampling.
    analyticalSampling()
        Abstract method for analytical sampling.
    appendDisplacement(displacement: np.ndarray, probability: float = None)
        Appends a displacement vector and its associated probability.
    """

    def __init__(self, uncertaintyModel: AbstractUncertaintyModel):
        super().__init__()
        self.UncertaintyModel = uncertaintyModel
        self._generatedDisplacements : list = []
        self._generatedProbabilities : list = []

    @property
    def generatedDisplacements(self) -> np.ndarray:
        return np.array(self._generatedDisplacements)

    @property
    def generatedProbabilities(self) -> np.ndarray:
        arr = np.array(self._generatedProbabilities)
        # Normalize probabilities to sum to 1
        return arr / np.sum(arr)

    @abstractmethod
    def MCsampling(self):
        pass

    @abstractmethod
    def analyticalSampling(self):
        pass

    def appendDisplacement(self, displacement: np.ndarray, probability: float = None):
        self._generatedDisplacements.append(displacement)
        if probability is not None:
            self._generatedProbabilities.append(probability)
        else:
            # If no probability is provided, equiprobable sampling
            self._generatedProbabilities.append(1.0)
