import numpy as np
from abc import ABC, abstractmethod

class AbstractsamplingMethod(ABC):
    """
    Abstract base class for sampling methods.

    Attributes
    ----------
    name : str
        The name of the sampling method.
    parameters : dict
        A dictionary to hold method parameters.
    """

    def __init__(self):
        super().__init__()
        self.name: str = "AbstractsamplingMethod"
        self.probabilities: np.ndarray = None
        self.displacements: np.ndarray = None

    @abstractmethod
    def sample(self):
        pass