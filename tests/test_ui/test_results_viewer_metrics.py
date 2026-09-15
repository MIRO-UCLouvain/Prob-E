import pytest

from Probabilistic_Evaluation.ui.resultsViewer import _with_metrics


def _row(success):
    return {"roi_name": "PTV", "clinical_goal": "Dx", "nominal": "50Gy", "nominal_passed": True, "success": success}


def test_with_metrics_matches_the_ported_js_compute():
    """Same numbers the old client-side `compute()` produced for this input."""
    weights = [0.6, 0.4]
    rows = [_row([True, True]), _row([True, False]), _row([False, True])]

    out = _with_metrics(rows, weights, prob_mass=1.0)

    assert [r["pr"] for r in out] == pytest.approx([1.0, 0.6, 0.4])
    assert [r["prg"] for r in out] == pytest.approx([1.0, 0.6, 0.4])
    # cpr is the running (weighted, logical-AND) joint probability, in row order
    assert [r["cpr"] for r in out] == pytest.approx([1.0, 0.6, 0.0])


def test_with_metrics_cpr_depends_on_row_order():
    """Reordering rows changes the cumulative joint probability (`cpr`), not `pr`/`prg`."""
    weights = [0.6, 0.4]
    row_b = _row([True, False])
    row_c = _row([False, True])

    forward = _with_metrics([row_b, row_c], weights, prob_mass=1.0)
    reversed_ = _with_metrics([row_c, row_b], weights, prob_mass=1.0)

    # own pass probability is unaffected by order
    assert forward[0]["pr"] == pytest.approx(reversed_[1]["pr"])
    assert forward[1]["pr"] == pytest.approx(reversed_[0]["pr"])

    # but the joint (cumulative) probability is 0 as soon as two disjoint successes combine,
    # regardless of which one comes first
    assert forward[1]["cpr"] == pytest.approx(0.0)
    assert reversed_[1]["cpr"] == pytest.approx(0.0)


def test_with_metrics_handles_zero_probability_mass():
    out = _with_metrics([_row([True])], weights=[0.0], prob_mass=0.0)
    assert out[0]["pr"] == 0.0
    assert out[0]["cpr"] == 0.0
