import pytest
from Probabilistic_Evaluation.data import DVH
import numpy as np


@pytest.fixture
def sample_dvh():
    dosemap = np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]])
    mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    max_DVH = 70
    spacing = (1.0, 1.0, 1.0)
    return DVH(dosemap=dosemap, mask=mask, max_DVH=max_DVH, spacing=spacing)


@pytest.fixture
def easy_dvh():
    arr = (np.arange(1000) + 1).reshape((10, 10, 10))
    mask = np.ones((10, 10, 10))
    return DVH(dosemap=arr, mask=mask, max_DVH=1010, spacing=(1.0, 1.0, 1.0))


def test_dvh_initialization(sample_dvh):
    dvh = sample_dvh
    assert dvh.dosemap is None
    assert dvh.mask.shape == (2, 2, 2)
    assert dvh.maxDVH == 70
    assert dvh.spacing == (1.0, 1.0, 1.0)
    assert dvh.dvh is not None
    assert dvh.bin_dose is not None


def test_dvh_invalid_maxDVH():
    dosemap = np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]])
    mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    spacing = (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        DVH(dosemap=dosemap, mask=mask, max_DVH=-10, spacing=spacing)


def test_dvh_mask_shape_mismatch():
    dosemap = np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]])
    mask = np.array([[[1, 0], [0, 1]]])  # Incorrect shape
    spacing = (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        DVH(dosemap=dosemap, mask=mask, max_DVH=70, spacing=spacing)


def test_dvh_spacing_invalid():
    dosemap = np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]])
    mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    with pytest.raises(ValueError):
        DVH(dosemap=dosemap, mask=mask, max_DVH=70, spacing=(1.0, -1.0, 1.0))
    with pytest.raises(ValueError):
        DVH(dosemap=dosemap, mask=mask, max_DVH=70, spacing=(1.0, 1.0))


@pytest.mark.parametrize(
    "dosemap,expected_DMax,expected_DMin,expected_DMean",
    [
        (np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]]), 50, 0, 30),
        (np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]), 0, 0, 0),
        (np.array([[[5, 5], [5, 5]], [[5, 5], [5, 5]]]), 5, 5, 5),
    ],
)
def test_dvh_statistics(dosemap, expected_DMax, expected_DMin, expected_DMean):
    mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    dvh = DVH(dosemap=dosemap, mask=mask, max_DVH=10, spacing=(1.0, 1.0, 1.0))
    assert dvh.DMax == expected_DMax
    assert dvh.DMin == expected_DMin
    assert dvh.DMean == expected_DMean


def test_dvh_dosemap_setter(sample_dvh):
    dvh = sample_dvh
    old_dvh = sample_dvh.dvh
    new_dosemap = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dvh.dosemap = new_dosemap
    assert dvh.dvh is not None
    assert dvh.dosemap is None
    assert not np.array_equal(dvh.dvh, old_dvh)


def test_easy_dvh_statistics(easy_dvh):
    dvh = easy_dvh
    assert dvh.DMin == 1
    assert dvh.DMax == 1000
    assert dvh.DMean == 500.5


def test_easy_dvh_dvh_computation(easy_dvh):
    dvh = easy_dvh
    bin_dose = dvh.bin_dose
    dvh_values = dvh.dvh
    dosemap = (np.arange(1000) + 1).reshape((10, 10, 10))
    # Check that the DVH values are correctly computed
    for i in range(len(bin_dose) - 1):
        dose_threshold = bin_dose[i]
        expected_volume = np.sum((dosemap >= dose_threshold)) / 1000 * 100
        assert np.isclose(dvh_values[i], expected_volume, atol=1)
    assert dvh_values[0] == 100  # At dose 0, full volume
    assert dvh_values[-1] == 0  # At max dose, no volume (if max_DVH > max dose in dosemap)


@pytest.mark.parametrize("dose,expected_vx", [
    (100, 90.1),
    (500, 50.0),
    (900, 10.0),
    (1000, 0.0),
])
def test_easy_dvh_Vx(easy_dvh, dose, expected_vx):
    dvh = easy_dvh
    value = dvh.computeVx(dose)
    assert np.isclose(value, expected_vx, atol=1)
    assert 0.0 <= value <= 100.0


@pytest.mark.parametrize("volume,expected_dx", [
    (1,1),
    (0.5,500),
    (0.1,900),
    (0.001,1000),
])
def test_easy_dvh_Dx(easy_dvh, volume, expected_dx):
    dvh = easy_dvh
    value = dvh.computeDx(volume)
    assert np.isclose(value, expected_dx, atol=1)
    assert 0.0 <= value <= 1000.0

@pytest.mark.parametrize("dose,expected_vcc", [
    (100, 0.901),
    (500, 0.500),
    (900, 0.099),
    (1000, 0),
])
def test_easy_dvh_Vcc(easy_dvh, dose, expected_vcc):
    dvh = easy_dvh
    value = dvh.computeVcc(dose)
    assert np.isclose(value, expected_vcc, atol=1)
    assert 0.0 <= value <= 1000.0

@pytest.mark.parametrize("volume,expected_dcc", [
    (1,1),
    (0.5,500),
    (0.1,900),
    (0.001,1000),
])
def test_easy_dvh_Dcc(easy_dvh, volume, expected_dcc):
    dvh = easy_dvh
    value = dvh.computeDcc(volume)
    assert np.isclose(value, expected_dcc, atol=1)
    assert 0.0 <= value <= 1000.0

def test_dvh_invalid_Dx_Vx(easy_dvh):
    dvh = easy_dvh
    with pytest.raises(ValueError):
        dvh.computeDx(-10)
    with pytest.raises(ValueError):
        dvh.computeDx(150)
    with pytest.raises(ValueError):
        dvh.computeVx(-5)

def test_dvh_invalid_Dcc_Vcc(easy_dvh):
    dvh = easy_dvh
    with pytest.raises(ValueError):
        dvh.computeDcc(-10)
    with pytest.raises(ValueError):
        dvh.computeVcc(-5)
    with pytest.raises(ValueError):
        dvh.computeDcc(2000)
