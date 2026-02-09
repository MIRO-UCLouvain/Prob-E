import pytest
import numpy as np
from Probabilistic_Evaluation.data.uncertaintyModel import Gaussian3DUncertaintyModel
from scipy.special import erf

@pytest.fixture
def simple_model():
    return Gaussian3DUncertaintyModel()

def test_initialization(simple_model):
    assert simple_model.name == "Gaussian3DUncertaintyModel"
    expected_params = {'mu_x': 0, 'mu_y': 0, 'mu_z': 0, 'sigma_x': 5 / (3.2), 'sigma_y': 5 / (3.2), 'sigma_z': 5 / (3.2)}
    assert simple_model.parameters == expected_params

def test_pdf_at_mean(simple_model):
    mu = (simple_model.parameters['mu_x'], simple_model.parameters['mu_y'], simple_model.parameters['mu_z'])
    pdf_value = simple_model.pdf(mu)
    expected_pdf = 1 / ((2 * np.pi) ** 1.5 *
                        simple_model.parameters['sigma_x'] *
                        simple_model.parameters['sigma_y'] *
                        simple_model.parameters['sigma_z'])
    assert np.isclose(pdf_value, expected_pdf), f"PDF at mean should be {expected_pdf}, got {pdf_value}"

def test_pdf_symmetry(simple_model):
    mu = (simple_model.parameters['mu_x'], simple_model.parameters['mu_y'], simple_model.parameters['mu_z'])
    offset = 1.0
    point1 = (mu[0] + offset, mu[1], mu[2])
    point2 = (mu[0] - offset, mu[1], mu[2])
    pdf1 = simple_model.pdf(point1)
    pdf2 = simple_model.pdf(point2)
    assert np.isclose(pdf1, pdf2), f"PDF should be symmetric around the mean, got {pdf1} and {pdf2}"

def test_cdf_bounds(simple_model):
    low_point = (-1e6, -1e6, -1e6)
    high_point = (1e6, 1e6, 1e6)
    cdf_low = simple_model.cdf(low_point)
    cdf_high = simple_model.cdf(high_point)
    assert np.isclose(cdf_low, 0.0), f"CDF at very low point should be close to 0, got {cdf_low}"
    assert np.isclose(cdf_high, 1.0), f"CDF at very high point should be close to 1, got {cdf_high}"

def test_cdf_monotonicity(simple_model):
    point1 = (0, 0, 0)
    point2 = (1, 1, 1)
    cdf1 = simple_model.cdf(point1)
    cdf2 = simple_model.cdf(point2)
    assert cdf2 >= cdf1, f"CDF should be non-decreasing, got {cdf1} and {cdf2}"

def test_sampled_points_pdf(simple_model):
    samples = simple_model.sample(1000)
    assert len(samples) == 1000, f"Number of sampled points should be 1000, got {len(samples)}"
    for point in samples:
        pdf_value = simple_model.pdf(point)
        assert pdf_value >= 0, f"PDF value should be non-negative, got {pdf_value} for point {point}"

def test_bounded_integral(simple_model):
    # use integral separation in each dimension and compute analytically
    # assume mu = 0 for all dimensions for simplicity
    Ix = 0.5 * (1 + erf(1 / (simple_model.parameters['sigma_x'] * np.sqrt(2)))) - \
         0.5 * (1 + erf(-1 / (simple_model.parameters['sigma_x'] * np.sqrt(2))))
    Iy = 0.5 * (1 + erf(1 / (simple_model.parameters['sigma_y'] * np.sqrt(2)))) - \
         0.5 * (1 + erf(-1 / (simple_model.parameters['sigma_y'] * np.sqrt(2))))
    Iz = 0.5 * (1 + erf(1 / (simple_model.parameters['sigma_z'] * np.sqrt(2)))) - \
         0.5 * (1 + erf(-1 / (simple_model.parameters['sigma_z'] * np.sqrt(2))))
    integral = Ix * Iy * Iz
    assert 0 < integral < 1, f"Integral over bounded region should be between 0 and 1, got {integral}"
    computed_integral = simple_model.boundedIntegral([-1, -1, -1], [1, 1, 1])
    assert np.isclose(integral, computed_integral), f"Computed integral should match expected value, got {computed_integral}"
    simple_model.parameters['mu_x'] = 1
    simple_model.parameters['mu_y'] = 1
    simple_model.parameters['mu_z'] = 1
    with pytest.raises(NotImplementedError):
        simple_model.boundedIntegral([-1, -1, -1], [1, 1, 1])