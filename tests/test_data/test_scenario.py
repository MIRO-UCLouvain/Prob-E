import numpy as np
import pytest
import gc
import weakref
from Probabilistic_Evaluation.data import Scenario


@pytest.fixture
def scenario():
    displacement = np.array([1, 2, 3])
    probability = 0.5
    return Scenario(displacement=displacement, probability=probability)


def test_dose_image_is_set_correctly(scenario):
    dose_image = np.random.rand(10, 10, 10)
    scenario.doseImage = dose_image
    assert np.array_equal(scenario.doseImage, dose_image)


def test_delete_dose_image_sets_internal_reference_to_none(scenario):
    arr = np.random.rand(4, 4, 4)
    scenario.doseImage = arr
    scenario.delete_doseImage()
    assert scenario.doseImage is None


def test_delete_dose_image_does_not_delete_external_reference(scenario):
    arr = np.random.rand(4, 4, 4)
    wref = weakref.ref(arr)
    scenario.doseImage = arr

    scenario.delete_doseImage()
    assert scenario.doseImage is None
    assert wref() is arr  # external reference still alive

    del arr
    gc.collect()
    assert wref() is None


def test_displacement_is_set_and_retrieved_correctly(scenario):
    expected_displacement = np.array([1, 2, 3])
    assert np.array_equal(scenario.displacement, expected_displacement)
    new_displacement = np.array([4, 5, 6])
    assert not np.array_equal(scenario.displacement, new_displacement)
    scenario.displacement = new_displacement
    assert np.array_equal(scenario.displacement, new_displacement)


def test_probability_is_set_correctly(scenario):
    assert scenario.probability == 0.5
    scenario.probability = 0.7
    assert scenario.probability == 0.7


def test_probability_raises_error_for_invalid_values(scenario):
    with pytest.raises(ValueError):
        scenario.probability = -0.1
    with pytest.raises(ValueError):
        scenario.probability = 1.1


def test_compute_shifted_image_updates_dose_image():
    initial_dose_image = np.random.rand(10, 10, 10)
    displacement = np.array([1, 1, 1])
    scenario = Scenario(displacement=displacement, probability=0.5)
    scenario.compute_shifted_image(initial_dose_image, displacement)
    assert scenario.doseImage is not None
    expected_shifted_image = np.roll(initial_dose_image, shift=(1, 1, 1), axis=(0, 1, 2))
    assert np.array_equal(scenario.doseImage, expected_shifted_image)
    assert not np.array_equal(scenario.doseImage, initial_dose_image)


def test_compute_shifted_image_with_zero_displacement():
    initial_dose_image = np.random.rand(10, 10, 10)
    displacement = np.array([0, 0, 0])
    scenario = Scenario(displacement=displacement, probability=0.5)
    scenario.compute_shifted_image(initial_dose_image, displacement)
    assert scenario.doseImage is not None
    assert np.array_equal(scenario.doseImage, initial_dose_image)


def test_compute_shifted_image_with_negative_displacement():
    initial_dose_image = np.random.rand(10, 10, 10)
    displacement = np.array([-1, -1, -1])
    scenario = Scenario(displacement=displacement, probability=0.5)
    scenario.compute_shifted_image(initial_dose_image, displacement)
    assert scenario.doseImage is not None
    expected_shifted_image = np.roll(initial_dose_image, shift=(-1, -1, -1), axis=(0, 1, 2))
    assert np.array_equal(scenario.doseImage, expected_shifted_image)
    assert not np.array_equal(scenario.doseImage, initial_dose_image)


def test_compute_shifted_image_non_matching_dimensions():
    initial_dose_image = np.random.rand(10, 10, 10)
    displacement = np.array([1, 1])  # Invalid displacement
    scenario = Scenario(displacement=displacement, probability=0.5)
    with pytest.raises(ValueError):
        scenario.compute_shifted_image(initial_dose_image, displacement)
    scenario.discplacement = np.array([1, 1, 1, 1])  # Invalid displacement
    with pytest.raises(ValueError):
        scenario.compute_shifted_image(initial_dose_image, displacement)
    scenario.displacement = np.array([1])
    with pytest.raises(ValueError):
        scenario.compute_shifted_image(initial_dose_image, displacement)
