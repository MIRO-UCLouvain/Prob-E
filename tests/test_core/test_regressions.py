"""Regression tests for the defects fixed after the September 2026 audit.

Each test is small enough to run in a few seconds on a laptop; the synthetic dose is a
30^3 sphere with a 1 mm penumbra.
"""

import logging

import numpy as np
import pytest

from Probabilistic_Evaluation.core.probabilisticEvaluator import ProbabilisticEvaluator
from Probabilistic_Evaluation.core.sampling.voronoiSampling import VoronoiSampling
from Probabilistic_Evaluation.data import PatientData, ROIContour
from Probabilistic_Evaluation.data.clinicalGoals import DCCClinicalGoal, DMaxClinicalGoal, DMeanClinicalGoal, DXClinicalGoal
from Probabilistic_Evaluation.data.uncertaintyModel._Gaussian3D import Gaussian3DUncertaintyModel
from Probabilistic_Evaluation.utils import get_partial_volume_mask, resampling_grid, shift_dose_image

logging.getLogger("ProbEval").addHandler(logging.NullHandler())


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synthetic():
    g = np.indices((30, 30, 30)).astype(float) - 14.5
    r = np.sqrt((g**2).sum(0))
    dose = (60.0 / (1.0 + np.exp((r - 8.0) / 1.0))).astype(np.float32)
    target = (r <= 6).astype(np.float32)
    oar = ((g[0] > 9) & (g[0] < 13) & (np.abs(g[1]) < 3) & (np.abs(g[2]) < 3)).astype(np.float32)
    return dose, {"TGT": target, "OAR": oar}


def _patient(synthetic, order):
    dose, masks = synthetic
    pd = PatientData(dose, masks, np.array([1.0, 1.0, 1.0]), patientID="synthetic")
    goals = []
    for kind in order:
        if kind == "np_oar":
            goals.append(DMeanClinicalGoal(20.0, masks["OAR"], "OAR", True, 0))
        elif kind == "p_tgt":
            g = DXClinicalGoal(50.0, masks["TGT"], "TGT", False, 0, volume=0.98)
            g.probabilistic = True
            goals.append(g)
        elif kind == "p_oar":
            g = DMaxClinicalGoal(20.0, masks["OAR"], "OAR", True, 0)
            g.probabilistic = True
            goals.append(g)
        elif kind == "p_bad":  # 5 cc requested on a 0.15 cc structure -> Dmin (no exception)
            g = DCCClinicalGoal(30.0, masks["OAR"], "OAR", True, 0, volume=5.0)
            g.probabilistic = True
            goals.append(g)
    pd.clinicalGoalsList = goals
    return pd


@pytest.fixture(scope="module")
def sampler():
    model = Gaussian3DUncertaintyModel(marginSize=3, simulateRandom=False)
    return VoronoiSampling(model, np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]), probabilityMass=0.99, enhancedSampling=False)


# ----------------------------------------------------------------------------- C2
def test_passing_rates_follow_goal_identity_not_position(synthetic, sampler):
    """Rates of the probabilistic goals must not depend on where non-probabilistic goals sit in the list."""
    t_first = ProbabilisticEvaluator(_patient(synthetic, ["p_tgt", "p_oar", "np_oar"]), sampler, nThreads=2).evaluate()
    t_mixed = ProbabilisticEvaluator(_patient(synthetic, ["np_oar", "p_tgt", "p_oar"]), sampler, nThreads=2).evaluate()

    def rates(table):
        prob = table[table["Probabilistic Objective"] == True]  # noqa: E712 - pandas boolean filter
        return dict(zip(prob["Clinical Goal"], prob["Passing Rate"]))

    assert rates(t_first) == rates(t_mixed)
    assert len(rates(t_mixed)) == 2
    assert t_mixed.iloc[0]["Passing Rate"] == "N/A"  # the non-probabilistic goal keeps its placeholder


def test_nominal_value_is_numeric(synthetic, sampler):
    table = ProbabilisticEvaluator(_patient(synthetic, ["p_tgt", "np_oar"]), sampler, nThreads=2).evaluate()
    assert all(isinstance(v, float) for v in table["Nominal Value"])


# ----------------------------------------------------------------------------- C3
def test_worker_exception_is_raised_not_swallowed(synthetic, sampler, monkeypatch):
    pd = _patient(synthetic, ["p_tgt"])
    evaluator = ProbabilisticEvaluator(pd, sampler, nThreads=2)

    def boom(*args, **kwargs):
        raise ZeroDivisionError("synthetic failure inside a scenario thread")

    monkeypatch.setattr(pd.clinicalGoalsList[0], "compute", boom)
    with pytest.raises(RuntimeError, match="scenarios failed") as excinfo:
        evaluator.evaluate()
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_unreachable_absolute_volume_does_not_crash_the_run(synthetic, sampler):
    """M4: D5cc on a 0.15 cc structure evaluates to Dmin in every scenario instead of raising."""
    table = ProbabilisticEvaluator(_patient(synthetic, ["p_bad"]), sampler, nThreads=2).evaluate()
    assert 0.0 <= table.iloc[0]["Passing Rate"] <= 1.0
    assert not np.isnan(table.iloc[0]["Passing Rate"])


# ----------------------------------------------------------------------------- C4
def test_voxelwise_min_max_are_thread_safe(synthetic):
    dose, masks = synthetic
    model = Gaussian3DUncertaintyModel(marginSize=3, simulateRandom=False)
    full = VoronoiSampling(model, np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]), probabilityMass=None, enhancedSampling=False)
    ref_min, ref_max = dose.copy(), np.zeros_like(dose)
    for point in full.voronoiPoints:
        shifted = shift_dose_image(dose, tuple(point))
        ref_min, ref_max = np.minimum(ref_min, shifted), np.maximum(ref_max, shifted)
    for _ in range(3):
        ev = ProbabilisticEvaluator(_patient(synthetic, ["p_tgt"]), full, nThreads=16, computeVWMin=True, computeVWMax=True)
        ev.evaluate()
        assert np.array_equal(ev.VWMin, ref_min)
        assert np.array_equal(ev.VWMax, ref_max)


# ----------------------------------------------------------------------------- H1
@pytest.mark.parametrize("spacing", [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 2.0]), np.array([2.0, 2.0, 2.0])])
def test_monte_carlo_probabilities_match_analytical_for_any_spacing(spacing):
    model = Gaussian3DUncertaintyModel(marginSize=3, n=28, simulateRandom=True)
    sampler = VoronoiSampling(model, np.array([6.0, 6.0, 6.0]), spacing, probabilityMass=None, enhancedSampling=False)
    points = sampler.voronoiPoints
    analytical = sampler._computeVoronoiProbabilitiesAnalytical(points, spacing)
    np.random.seed(0)
    mc = sampler._computeVoronoiProbabilitiesMC(points, spacing, num_samples=400_000)
    for p in ([0, 0, 0], [1, 0, 0], [0, 0, 1]):
        sel = np.all(points == p, axis=1)
        assert np.isclose(mc[sel][0], analytical[sel][0], atol=2e-3), (spacing, p)


# ----------------------------------------------------------------------------- H2
def test_symmetry_classes_respect_anisotropic_sigma_and_spacing():
    iso = Gaussian3DUncertaintyModel(marginSize=3, simulateRandom=False)
    s = VoronoiSampling(iso, np.array([3.0, 3.0, 3.0]), np.array([1.0, 1.0, 1.0]), probabilityMass=None, enhancedSampling=False)
    assert s._symmetry_key([1, 0, 0]) == s._symmetry_key([0, 0, -1])  # permutation + sign flip
    assert s._symmetry_key([2, 1, 0]) == s._symmetry_key([0, -1, 2])
    assert s._symmetry_key([3, 0, 0]) != s._symmetry_key([2, 2, 1])  # same norm, not equivalent

    aniso = Gaussian3DUncertaintyModel(manual_input={"sys_parameters": {"mu_x": 0, "mu_y": 0, "mu_z": 0, "sigma_x": 1.0, "sigma_y": 1.0, "sigma_z": 2.0}, "rand_parameters": None})
    s = VoronoiSampling(aniso, np.array([3.0, 3.0, 3.0]), np.array([1.0, 1.0, 1.0]), probabilityMass=None, enhancedSampling=False)
    assert s._symmetry_key([1, 0, 0]) == s._symmetry_key([0, 1, 0])  # x and y interchangeable
    assert s._symmetry_key([1, 0, 0]) != s._symmetry_key([0, 0, 1])  # z has a different sigma

    s = VoronoiSampling(iso, np.array([3.0, 3.0, 3.0]), np.array([1.0, 1.0, 2.0]), probabilityMass=None, enhancedSampling=False)
    assert s._symmetry_key([1, 0, 0]) != s._symmetry_key([0, 0, 1])  # z has a different spacing


def test_reduction_averages_only_over_symmetry_classes():
    aniso = Gaussian3DUncertaintyModel(manual_input={"sys_parameters": {"mu_x": 0, "mu_y": 0, "mu_z": 0, "sigma_x": 1.0, "sigma_y": 1.0, "sigma_z": 2.0}, "rand_parameters": None})
    np.random.seed(1)
    s = VoronoiSampling(aniso, np.array([4.0, 4.0, 4.0]), np.array([1.0, 1.0, 1.0]), probabilityMass=0.9, enhancedSampling=True)
    pts, pr = s.voronoiPoints, s.voronoiProbabilities

    def prob(p):
        sel = np.all(pts == p, axis=1)
        return pr[sel][0]

    assert prob([1, 0, 0]) == prob([0, -1, 0])  # equivalent cells share one averaged value
    assert prob([0, 0, 1]) > prob([1, 0, 0])  # larger sigma along z: more mass in the z neighbour cell


# ----------------------------------------------------------------------------- H4
def _sphere_contours(r, dz, npts=360):
    polys = []
    for z in np.arange(-r + 0.0001, r, dz):
        rz = np.sqrt(max(r * r - z * z, 0.0))
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        polys.append(np.column_stack([rz * np.cos(th), rz * np.sin(th), np.full(npts, z)]).ravel())
    return polys


@pytest.mark.parametrize("dz,tol", [(1.0, 0.02), (2.5, 0.04), (3.0, 0.05)])
def test_partial_volume_mask_volume_with_coarse_contours(dz, tol):
    r = 10.0
    contour = ROIContour("sphere", _sphere_contours(r, dz))
    mask = get_partial_volume_mask(contour, np.array([-20.0, -20.0, -20.0]), np.array([41, 41, 41]), np.array([1.0, 1.0, 1.0]), precision=16)
    exact = 4.0 / 3.0 * np.pi * r**3
    assert abs(mask.sum() / exact - 1.0) < tol


def test_between_contour_slices_are_distance_weighted():
    # two square contours 4 mm apart, 16 and 32 mm^2: the cross-section grows linearly in between, and each plane extends
    # half the plane distance beyond the ends, so the volume is (16 + 32) mm^2 * 4 mm for any voxel size
    lower = np.array([0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0], float)
    upper = np.array([0, 0, 4, 8, 0, 4, 8, 4, 4, 0, 4, 4], float)
    wedge = ROIContour("wedge", [lower, upper])
    mask = get_partial_volume_mask(wedge, np.array([-4.0, -4.0, -4.0]), np.array([16, 16, 12]), np.array([1.0, 1.0, 1.0]), precision=16)
    area = mask.sum(axis=(0, 1))  # mm^2 in each 1 mm slice
    k0, k1 = 4, 8  # slice indices of the two contours
    assert np.allclose(area[k0 + 1 : k1], [20.0, 24.0, 28.0], atol=0.05)
    for spacing in (0.5, 1.0, 2.0):  # in-plane edges fall on sub-voxel boundaries, so only z can change the volume
        size, origin = resampling_grid([16, 16, 12], [1.0, 1.0, 1.0], [-4.0, -4.0, -4.0], [spacing] * 3)
        mask = get_partial_volume_mask(wedge, origin, size, np.array([spacing] * 3), precision=16)
        assert np.isclose(mask.sum() * spacing**3, 192.0, rtol=0.005), spacing


# ----------------------------------------------------------------------------- mask / dose grid alignment
@pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (2.0, 2.0, 1.0)])
def test_partial_volume_mask_stays_on_the_contours_for_any_spacing(spacing):
    """Masks sit on the contours for any voxel size: in-plane around the grid origin taken as the centre of the first voxel
    (RayStation Corner + spacing/2), along z with every contour plane covering half the plane distance on either side.

    The rasterizer used to add 0.5 * (1 - spacing) mm in-plane, and to snap the planes and the ROI ends to whole voxels in z.
    """
    spacing = np.asarray(spacing, dtype=float)
    size, origin = resampling_grid([60, 60, 40], [1.0, 1.0, 1.0], [-30.0, -30.0, -20.0], spacing)  # as the DICOM reader builds it
    offsets, volumes = [], []
    for k in range(8):  # sub-voxel box positions, so that the partial-volume quantization averages out
        cx, cy = 0.37 + k * spacing[0] / 8, -0.61 + k * spacing[1] / 8
        box = [np.array([cx - 8.3, cy - 5.7, z, cx + 8.3, cy - 5.7, z, cx + 8.3, cy + 5.7, z, cx - 8.3, cy + 5.7, z]) for z in np.arange(-8.0, 8.5, 1.0)]
        mask = get_partial_volume_mask(ROIContour("box", box), origin, size, spacing, precision=16).astype(np.float64)
        index = np.indices(mask.shape).reshape(3, -1)
        centroid = origin + (index * mask.ravel()).sum(axis=1) / mask.sum() * spacing
        offsets.append(centroid - [cx, cy, 0.0])
        volumes.append(mask.sum() * np.prod(spacing))
    # partial-volume quantization scales with the voxel size; the old in-plane offset was 5 to 10 times this tolerance
    assert np.allclose(np.mean(offsets, axis=0), 0.0, atol=0.05 * spacing.max())
    assert np.isclose(np.mean(volumes), 16.6 * 11.4 * 17.0, rtol=0.01)  # 17 planes 1 mm apart


# ----------------------------------------------------------------------------- M14 / H6
def test_resampling_grid_keeps_the_voxel_edge_extent():
    size, origin = resampling_grid([100, 8, 8], [2.5, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert list(size) == [250, 8, 8]
    assert np.allclose(origin, [-0.75, 0.0, 0.0])  # outer edge -1.25 mm + half a new voxel
    size, origin = resampling_grid([10, 10, 10], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0], [3.0, 3.0, 3.0])
    assert list(size) == [10, 10, 10] and np.allclose(origin, [5.0, 5.0, 5.0])  # unchanged for equal spacing
    size, _ = resampling_grid([3, 3, 3], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    assert list(size) == [3, 3, 3]  # no extra voxel from floating point noise


# ----------------------------------------------------------------------------- M11 (editor round trip, no Streamlit needed)
def test_editor_keeps_probabilistic_flag_and_file_order(tmp_path):
    from Probabilistic_Evaluation.ui.clinicalGoalsEditor import _insert_in_roi_group, _sort_goals, load_goals, save_goals

    path = tmp_path / "goals.json"
    path.write_text(
        '[{"ROI": "Zeta", "type": "Dx", "dose": 50.4, "volume": 98, "lower_is_better": false, "probabilistic": true},\n'
        ' {"ROI": "Alpha", "type": "Dmean", "dose": 20.0, "lower_is_better": true},\n'
        ' {"ROI": "Zeta", "type": "Dmax", "dose": 55.0, "lower_is_better": true, "probabilistic": false}]',
        encoding="utf-8",
    )
    goals = load_goals(path)
    assert [g["probabilistic"] for g in goals] == [True, False, False]
    assert [g["ROI"] for g in _sort_goals(goals)] == ["Alpha", "Zeta", "Zeta"]  # display order only

    new = {"ROI": "Zeta", "base": "Dmin", "dose": 40.0, "volume": None, "in_cc": False, "lower_is_better": False, "priority": 0, "probabilistic": True, "extra": {}}
    _insert_in_roi_group(goals, new)
    save_goals(path, goals)
    reloaded = load_goals(path)
    assert [g["ROI"] for g in reloaded] == ["Zeta", "Alpha", "Zeta", "Zeta"]  # file order kept, new goal after its ROI group
    assert [g["probabilistic"] for g in reloaded] == [True, False, False, True]


# ----------------------------------------------------------------------------- M12 (viewer formatting)
def test_viewer_formats_nominal_values_with_the_goal_unit():
    from pathlib import Path

    from Probabilistic_Evaluation.ui.resultsViewer import build_payload

    table = [
        {"Mask Name": "Brain", "Clinical Goal": "V60.0Gy<=3.00cc", "Nominal Value": 0.42, "Nominal Success": True, "Probabilistic Objective": False},
        {"Mask Name": "GTV", "Clinical Goal": "D98.0%>=50.4", "Nominal Value": 50.3996, "Nominal Success": False, "Probabilistic Objective": False},
        {"Mask Name": "X", "Clinical Goal": "V50.0Gy>=95.00%", "Nominal Value": 0.9712, "Nominal Success": True, "Probabilistic Objective": False},
        {"Mask Name": "Y", "Clinical Goal": "D0.03cc<=45.0", "Nominal Value": "N/A", "Nominal Success": "N/A", "Probabilistic Objective": False},
    ]
    payload = build_payload({"x": {"data": {"Probability Array": [1.0], "Probability Mass": 1.0, "Table": table}, "path": Path("x.json")}})
    shown = [row["nominal"] for row in payload[0]["nominal_rows"]]
    assert shown == ["0.42cc", "50.40Gy", "97.12%", "N/A"]  # two decimals, unit from the goal
