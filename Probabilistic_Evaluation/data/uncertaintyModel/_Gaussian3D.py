import numpy as np
import scipy as sp
from scipy.special import erf

from ._abstractUncertaintyModel import AbstractUncertaintyModel


class Gaussian3DUncertaintyModel(AbstractUncertaintyModel):
    """
    A class to represent a 3D Gaussian uncertainty model.
    ! ASSUMING INDEPENDENT DIMENSIONS !

    Attributes
    ----------
    name : str (default: "Gaussian3DUncertaintyModel")
        The name of the uncertainty model.
    parameters : dict
        A dictionary containing the parameters of the Gaussian model:
        - 'mu_x': Mean in x direction (default: 0)
        - 'mu_y': Mean in y direction (default: 0)
        - 'mu_z': Mean in z direction (default: 0)
        - 'sigma_x': Standard deviation in x direction (default: 5/3.2)
        - 'sigma_y': Standard deviation in y direction (default: 5/3.2)
        - 'sigma_z': Standard deviation in z direction (default: 5/3.2)
    """

    def __inti__(self, parameters: dict = {'mu_x': 0, 'mu_y': 0, 'mu_z': 0, 'sigma_x': 5 / (3.2), 'sigma_y': 5 / (3.2),
                                           'sigma_z': 5 / (3.2)}):
        super().__init__()
        self.name: str = "Gaussian3DUncertaintyModel"
        self.parameters: dict = parameters

    def pdf(self, x):
        """
        Compute the probability density function at point x.

        Parameters
        ----------
        x : array_like
            A 3D point (x, y, z) at which to evaluate the PDF.

        Returns
        -------
        float
            The value of the PDF at point x.
        """
        mu_x = self.parameters['mu_x']
        mu_y = self.parameters['mu_y']
        mu_z = self.parameters['mu_z']
        sigma_x = self.parameters['sigma_x']
        sigma_y = self.parameters['sigma_y']
        sigma_z = self.parameters['sigma_z']

        coeff = 1 / ((2 * np.pi) ** 1.5 * sigma_x * sigma_y * sigma_z)
        exponent = -(((x[0] - mu_x) ** 2) / (2 * sigma_x ** 2) +
                     ((x[1] - mu_y) ** 2) / (2 * sigma_y ** 2) +
                     ((x[2] - mu_z) ** 2) / (2 * sigma_z ** 2))

        return coeff * np.exp(exponent)

    def cdf(self, x):
        """
        Compute the cumulative distribution function at point x.

        Parameters
        ----------
        x : array_like
            A 3D point (x, y, z) at which to evaluate the CDF.

        Returns
        -------
        float
            The value of the CDF at point x.
        """
        mu_x = self.parameters['mu_x']
        mu_y = self.parameters['mu_y']
        mu_z = self.parameters['mu_z']
        sigma_x = self.parameters['sigma_x']
        sigma_y = self.parameters['sigma_y']
        sigma_z = self.parameters['sigma_z']

        cdf_x = 0.5 * (1 + erf((x[0] - mu_x) / (sigma_x * np.sqrt(2))))
        cdf_y = 0.5 * (1 + erf((x[1] - mu_y) / (sigma_y * np.sqrt(2))))
        cdf_z = 0.5 * (1 + erf((x[2] - mu_z) / (sigma_z * np.sqrt(2))))

        return cdf_x * cdf_y * cdf_z

    def sample(self, n):
        """
        Generate n random samples from the 3D Gaussian uncertainty model.

        Parameters
        ----------
        n : int
            The number of samples to generate.

        Returns
        -------
        ndarray
            An array of shape (n, 3) containing n random samples.
        """
        mu_x = self.parameters['mu_x']
        mu_y = self.parameters['mu_y']
        mu_z = self.parameters['mu_z']
        sigma_x = self.parameters['sigma_x']
        sigma_y = self.parameters['sigma_y']
        sigma_z = self.parameters['sigma_z']

        mean = [mu_x, mu_y, mu_z]
        cov = [[sigma_x ** 2, 0, 0],
               [0, sigma_y ** 2, 0],
               [0, 0, sigma_z ** 2]]

        samples = np.random.multivariate_normal(mean, cov, n)
        return samples

    def boundedIntegral(self, a, b):
        """
        Compute the integral of the 3D Gaussian function over the bounded region defined by a and b.
        If the mean is zero in all dimensions, the integral is computed as the product of 1D integrals.

        Parameters
        ----------
        a : array_like
            The lower bounds (a_x, a_y, a_z).
        b : array_like
            The upper bounds (b_x, b_y, b_z).

        Returns
        -------
        float
            The value of the bounded integral over the specified region.
        """
        if self.parameters['mu_x'] == self.parameters['mu_y'] == self.parameters['mu_z'] == 0:
            integral_x = self.bounded1DIntegral(a[0], b[0], 0, self.parameters['sigma_x'])
            integral_y = self.bounded1DIntegral(a[1], b[1], 0, self.parameters['sigma_y'])
            integral_z = self.bounded1DIntegral(a[2], b[2], 0, self.parameters['sigma_z'])
            return integral_x * integral_y * integral_z
        else:
            raise NotImplementedError("Bounded integral for non-zero mean is not implemented yet.")

    def bounded1DIntegral(self, a, b, mu, sigma):
        """
        Compute the integral of a 1D Gaussian function from a to b.

        Parameters
        ----------
        a : float
            The lower bound.
        b : float
            The upper bound.
        mu : float
            The mean of the Gaussian.
        sigma : float
            The standard deviation of the Gaussian.
        """
        coeff = 0.5 * (erf((b - mu) / (sigma * np.sqrt(2))) - erf((a - mu) / (sigma * np.sqrt(2))))
        return coeff
