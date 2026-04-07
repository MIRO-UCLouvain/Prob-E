from abc import ABC, abstractmethod

class AbstractUncertaintyModel(ABC):
    """
    Abstract base class for uncertainty models.

    Attributes
    ----------
    name : str
        The name of the uncertainty model.
    sys_parameters : dict
        A dictionary to hold systematic setup error parameters.
    rand_parameters : dict
        A dictionary to hold random setup error parameters.
    """

    def __init__(self):
        super().__init__()
        self.name: str = "AbstractUncertaintyModel"
        self.sys_parameters: dict = {}
        self.rand_parameters: dict = {}

    @abstractmethod
    def pdf(self, x):
        """
        Compute the probability density function at point x.

        Parameters
        ----------
        x : float or array-like
            The point at which to evaluate the PDF.

        Returns
        -------
        float
            The value of the PDF at point x.

        """
        pass

    @abstractmethod
    def cdf(self, x):
        """
        Compute the cumulative distribution function at point x.

        Parameters
        ----------
        x : float
            The point at which to evaluate the CDF.

        Returns
        -------
        float
            The value of the CDF at point x.

        """
        pass

    @abstractmethod
    def sample(self, n):
        """
        Generate n random samples from the uncertainty model.

        Parameters
        ----------
        n : int
            The number of samples to generate.

        Returns
        -------
        list
            A list of n random samples.

        """
        pass