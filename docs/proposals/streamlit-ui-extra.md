# Proposal: ship the Streamlit UI as an optional extra

> **Status: implemented, then superseded.** The `ui` extra and its
> `pip install rt-probabilistic-evaluation[ui]` mechanics described below are
> still in place. The specific file layout this document describes
> (`clinicalGoalsEditor.py` under `io/`, `JSONreader.py` under `ui/`) was
> later reorganized — see
> [`streamlit-ui-split-and-aggrid-viewer.md`](streamlit-ui-split-and-aggrid-viewer.md)
> for the current layout: both tools now live under `ui/`
> (`clinicalGoalsEditor.py` and `resultsViewer.py`), sharing a
> `ui/_launch.py` helper, with the results viewer rewritten around
> `streamlit-aggrid`. This document is kept for the packaging/extras
> reasoning, which is still accurate.

## Goal

`pip install rt-probabilistic-evaluation` should install only the core
scientific stack (numpy, scipy, pydicom, pandas, matplotlib, scikit-image).
Anyone who also wants the interactive clinical-goals editor should be able to
opt in with:

```bash
pip install rt-probabilistic-evaluation[ui]
```

## Current state

- The only real Streamlit UI in the repo is
  `src/Probabilistic_Evaluation/io/clinicalGoalsEditor.py`
  (`edit_clinical_goals()`). It is already written defensively:
  - `import streamlit as st` happens **inside** `_run_app()`, never at module
    top level, so simply importing `Probabilistic_Evaluation` (or
    `Probabilistic_Evaluation.io`) never requires Streamlit to be installed.
  - `edit_clinical_goals()` explicitly checks
    `importlib.util.find_spec("streamlit")` and raises a clear
    `RuntimeError` if it's missing, instead of failing with an import
    traceback.
  - `io/__init__.py` does **not** import `clinicalGoalsEditor` at all today,
    so the module isn't even reachable as `Probabilistic_Evaluation.io.edit_clinical_goals` yet.
  - Net effect: Streamlit is already an optional *runtime* dependency in
    behavior, it's just not declared as one in packaging metadata, so
    `pip install rt-probabilistic-evaluation` currently gives no way to know
    the UI extra exists or to pull it in with one command.
- `streamlit_app/JSONreader.py` (top-level, outside `src/`) is a second,
  unrelated results viewer. Despite the folder's name it does **not** import
  `streamlit` — it's a plain `http.server`-based viewer. It was not part of
  the installable package (it lived outside `src/`) and was only importable
  from a repo checkout, via the test-only `pythonpath = ["."]` setting in
  `pyproject.toml`. See "Decision: `streamlit_app/JSONreader.py`" below for
  what changed.

## Proposed change

Add a `ui` group to `[project.optional-dependencies]` in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=7.4.0,<8.0",
  "ruff>=0.0.297,<1.0",
]
ui = [
  "streamlit>=1.42,<2.0",
]
```

Notes on the version floor: `clinicalGoalsEditor.py` uses
`st.selectbox(..., accept_new_options=True)`, a relatively recent Streamlit
parameter. `>=1.42` is a reasonable floor based on when that API surface
appeared, but it should be confirmed against whichever Streamlit version the
editor has actually been exercised with locally (`pip show streamlit` in a
working dev environment) before this is merged, and bumped if needed.

This is the whole packaging change for the clinical-goals editor — no source
code changes were required there, because the module already guards its
`streamlit` import.

## Decision: `streamlit_app/JSONreader.py`

The request was for this module to be installed "only when the user installs
the `ui` extra". That can't be done literally: pip extras only add extra
*dependencies* at install time, they can't make a single package's wheel
conditionally include or exclude files — `pip install pkg` and
`pip install pkg[ui]` install the exact same wheel, built once ahead of
time. Getting true file-level conditionality would require splitting this
module into a second, separately published distribution that `ui` depends
on — extra infrastructure (a second package to version and publish) for one
small module.

Instead this was resolved the same way `clinicalGoalsEditor.py` already
resolves it, gating at runtime rather than at install time:

- The module moved from the stray top-level `streamlit_app/` directory into
  the package proper, as `Probabilistic_Evaluation.ui.JSONreader`. It is now
  always installed and always importable, on an equal footing with `io`,
  `core` and `data`.
- Its one real entry point, `launch()`, now calls a guard
  (`_require_ui_extra()`) that checks `importlib.util.find_spec("streamlit")`
  and raises a `RuntimeError` telling the user to
  `pip install rt-probabilistic-evaluation[ui]` if it's missing — even though
  this module has no actual code-level need for Streamlit itself. The
  `streamlit` dependency is used purely as the marker for "the `ui` extra is
  installed", since it's the only thing that extra currently declares.
- The test that exercised it (`tests/test_utils/test_jsonreader_import.py`)
  now imports `Probabilistic_Evaluation.ui.JSONreader` directly, and the
  test-only `pythonpath = ["."]` pytest setting was removed since nothing
  needs it anymore.

Net effect for users: `pip install rt-probabilistic-evaluation` and
`pip install rt-probabilistic-evaluation[ui]` install the same files, but
only the latter lets `Probabilistic_Evaluation.ui.JSONreader.launch()` (and
`edit_clinical_goals()`) actually run instead of raising immediately.

## Follow-up (all implemented)

1. **Expose the editor from the public API.** `io/__init__.py` now does
   `from .clinicalGoalsEditor import edit_clinical_goals` and lists it in
   `__all__`, so `from Probabilistic_Evaluation.io import edit_clinical_goals`
   works without reaching into the private module path. This was safe to do
   unconditionally since it introduces no top-level `streamlit` import.
2. **Document the extra in `README.md`.** Added an "Optional: clinical goals
   editor UI" section with the `[ui]` install command and both a Python and
   a shell usage example.
3. **CI now also installs `[ui]`.** `.github/workflows/testing_pipeline.yml`
   runs the test job over a matrix of `extras: ["dev", "dev,ui"]`: the
   `"dev"` leg (streamlit absent) guards against an eager `import streamlit`
   regression, the `"dev,ui"` leg confirms the ui-enabled path also passes.
4. **Convenience entry point added.** `rt-eval-goals`, defined under
   `[project.scripts]` as `Probabilistic_Evaluation.io.clinicalGoalsEditor:_cli`,
   lets a `[ui]` install run `rt-eval-goals goals.json` from a shell. The
   script is always installed (console-script entries can't be conditioned
   on an extra either), but `edit_clinical_goals()` still raises its usual
   clear error if `streamlit` isn't present.

## Non-goals

- No change to the required (non-optional) `dependencies` list.
- No renaming of `rt-probabilistic-evaluation` or its `[dev]` extra.
