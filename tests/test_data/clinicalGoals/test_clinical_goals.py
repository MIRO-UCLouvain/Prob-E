import pytest
import numpy as np
from Probabilistic_Evaluation.data.clinicalGoals import *


@pytest.fixture
def goal_classes():
    return [
        DCCClinicalGoal,
        DXClinicalGoal,
        VXClinicalGoal,
        VCCClinicalGoal,
        DMaxClinicalGoal,
        DMeanClinicalGoal,
        DMinClinicalGoal,
    ]


@pytest.fixture
def example_parameters():
    return {
        "prescription": 50.0,
        "prescription_volume": 0.95,
        "maskName": "PTV",
        "mask": np.ones((10, 10, 10)),  # Example mask
    }


def test_clinical_goals_loaded(goal_classes):
    for goal in goal_classes:
        cls_name = goal.__name__
        assert hasattr(__import__('Probabilistic_Evaluation.data.clinicalGoals', fromlist=[cls_name]),
                       cls_name), f"{cls_name} not found in clinicalGoals module"


def test_clinical_goals_instantiation(goal_classes, example_parameters):
    # Example parameters for instantiation
    prescription = example_parameters["prescription"]
    prescription_volume = example_parameters["prescription_volume"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    for goal_class in goal_classes:
        if goal_class in [DCCClinicalGoal, DXClinicalGoal]:
            goal_instance = goal_class(prescription, mask, maskName, volume=0.95)
            assert goal_instance.prescription == prescription, f"Prescription mismatch for {goal_class.__name__}"
        elif goal_class in [VXClinicalGoal, VCCClinicalGoal]:
            goal_instance = goal_class(prescription_volume, mask, maskName, dose=30.0)
            assert goal_instance.prescription == prescription_volume, f"Prescription mismatch for {goal_class.__name__}"
        else:
            goal_instance = goal_class(prescription, mask, maskName)
            assert goal_instance.prescription == prescription, f"Prescription mismatch for {goal_class.__name__}"

        assert goal_instance.maskName == maskName
        assert goal_instance.mask is mask
        if goal_class is DMinClinicalGoal:
            assert goal_instance.lower_is_better is False  # specificfor DMinClinicalGoal
        else:
            assert goal_instance.lower_is_better is True  # default value

        assert isinstance(goal_instance, goal_class), f"Failed to instantiate {goal_class.__name__}"


# python
def test_dcc_clinical_goal_parameters(example_parameters):
    prescription = example_parameters["prescription"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = DCCClinicalGoal(prescription, mask, maskName, volume=95.0)
    assert isinstance(goal, DCCClinicalGoal)
    assert goal.volume == 95.0
    with pytest.raises(ValueError):
        DCCClinicalGoal(prescription, mask, maskName, volume=-5.0)
    with pytest.raises(ValueError):
        DCCClinicalGoal(prescription, mask, maskName)


def test_dx_clinical_goal_parameters(example_parameters):
    prescription = example_parameters["prescription"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = DXClinicalGoal(prescription, mask, maskName, volume=0.95)
    assert isinstance(goal, DXClinicalGoal)
    assert goal.volume == 0.95
    with pytest.raises(ValueError):
        DXClinicalGoal(prescription, mask, maskName, volume=-0.1)
    with pytest.raises(ValueError):
        DXClinicalGoal(prescription, mask, maskName, volume=1.5)
    with pytest.raises(ValueError):
        DXClinicalGoal(prescription, mask, maskName)


def test_vx_clinical_goal_parameters(example_parameters):
    prescription = example_parameters["prescription_volume"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = VXClinicalGoal(prescription, mask, maskName, dose=30.0)
    assert isinstance(goal, VXClinicalGoal)
    assert goal.dose == 30.0
    with pytest.raises(ValueError):
        VXClinicalGoal(prescription, mask, maskName, dose=-10.0)
    with pytest.raises(ValueError):
        VXClinicalGoal(prescription, mask, maskName)


def test_vcc_clinical_goal_parameters(example_parameters):
    prescription = example_parameters["prescription_volume"] * 1000
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = VCCClinicalGoal(prescription, mask, maskName, dose=30.0)
    assert isinstance(goal, VCCClinicalGoal)
    assert goal.dose == 30.0
    with pytest.raises(ValueError):
        VCCClinicalGoal(prescription, mask, maskName, dose=-10.0)
    with pytest.raises(ValueError):
        VCCClinicalGoal(prescription, mask, maskName)


def test_dmax_clinical_goal_instantiation(example_parameters):
    prescription = example_parameters["prescription"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = DMaxClinicalGoal(prescription, mask, maskName)
    assert isinstance(goal, DMaxClinicalGoal)


def test_dmean_clinical_goal_instantiation(example_parameters):
    prescription = example_parameters["prescription"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = DMeanClinicalGoal(prescription, mask, maskName)
    assert isinstance(goal, DMeanClinicalGoal)


def test_dmin_clinical_goal_instantiation(example_parameters):
    prescription = example_parameters["prescription"]
    maskName = example_parameters["maskName"]
    mask = example_parameters["mask"]

    goal = DMinClinicalGoal(prescription, mask, maskName)
    assert isinstance(goal, DMinClinicalGoal)
