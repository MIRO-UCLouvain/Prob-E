import numpy as np
import pytest
from Probabilistic_Evaluation.data import PatientData
from Probabilistic_Evaluation.data.clinicalGoals import *

@pytest.fixture
def sample_patient_data():
    ct_image = np.random.rand(10, 10, 10)
    dose_image = np.random.rand(10, 10, 10)
    mask_dict = {
        "StructureA": np.random.randint(0, 2, size=(10, 10, 10)),
        "StructureB": np.random.randint(0, 2, size=(10, 10, 10))
    }
    return PatientData(ct_image, dose_image, mask_dict)

def test_initialization(sample_patient_data):
    patient_data = sample_patient_data
    assert patient_data.ctImage.shape == (10, 10, 10)
    assert patient_data.doseImage.shape == (10, 10, 10)
    assert "StructureA" in patient_data._maskDict
    assert "StructureB" in patient_data._maskDict
    assert patient_data.spacing == (1.0, 1.0, 1.0)

def test_ct_image_setter(sample_patient_data):
    patient_data = sample_patient_data
    new_ct_image = np.random.rand(10, 10, 10)
    patient_data.ctImage = new_ct_image
    assert np.array_equal(patient_data.ctImage, new_ct_image)
    with pytest.raises(ValueError):
        patient_data.ctImage = np.random.rand(5, 5, 5)

def test_dose_image_setter(sample_patient_data):
    patient_data = sample_patient_data
    new_dose_image = np.random.rand(10, 10, 10)
    patient_data.doseImage = new_dose_image
    assert np.array_equal(patient_data.doseImage, new_dose_image)
    with pytest.raises(ValueError):
        patient_data.doseImage = np.random.rand(5, 5, 5)

def test_mask_dict_setter(sample_patient_data):
    patient_data = sample_patient_data
    new_mask_dict = {
        "StructureC": np.random.randint(0, 2, size=(10, 10, 10))
    }
    patient_data.maskDict = new_mask_dict
    assert "StructureC" in patient_data.maskDict
    with pytest.raises(ValueError):
        patient_data.maskDict = {
            "StructureD": np.random.randint(0, 2, size=(5, 5, 5))
        }

def test_spacing_setter(sample_patient_data):
    patient_data = sample_patient_data
    new_spacing = (1.0, 1.0, 1.0)
    patient_data.spacing = new_spacing
    assert patient_data.spacing == new_spacing
    with pytest.raises(ValueError):
        patient_data.spacing = (1.0, -1.0, 1.0)

def test_clinical_goals_list(sample_patient_data):
    patient_data = sample_patient_data
    assert isinstance(patient_data.clinicalGoalsList, list)
    assert len(patient_data.clinicalGoalsList) == 0

def test_scenario_list(sample_patient_data):
    patient_data = sample_patient_data
    assert isinstance(patient_data.scenarioList, list)
    assert len(patient_data.scenarioList) == 0

def test_get_mask(sample_patient_data):
    patient_data = sample_patient_data
    mask = patient_data.getMask("StructureA")
    assert np.array_equal(mask, patient_data.maskDict["StructureA"])
    with pytest.raises(KeyError):
        patient_data.getMask("NonExistentStructure")

def test_get_clinical_goals_for_structure(sample_patient_data):
    patient_data = sample_patient_data
    with pytest.raises(KeyError):
        patient_data.getClinicalGoalsForStructure("StructureA")
    # Add a clinical goal for testing
    mask = patient_data.getMask("StructureA")
    clinical_goal = DMaxClinicalGoal(prescription=50.0, mask=mask, maskName="StructureA", priority=1)
    patient_data.clinicalGoalsList.append(clinical_goal)
    goals = patient_data.getClinicalGoalsForStructure("StructureA")
    assert len(goals) == 1
    assert goals[0] == clinical_goal

