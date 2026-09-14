from streamlit_app import JSONreader


def test_jsonreader_module_is_importable():
    assert hasattr(JSONreader, "load_results")
    assert hasattr(JSONreader, "build_payload")
