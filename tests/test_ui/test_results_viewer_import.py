from Probabilistic_Evaluation.ui import resultsViewer


def test_results_viewer_module_is_importable():
    assert hasattr(resultsViewer, "load_results")
    assert hasattr(resultsViewer, "build_payload")
    assert hasattr(resultsViewer, "view_results")
