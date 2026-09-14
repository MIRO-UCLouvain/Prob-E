# Plan: split `ui` into two Streamlit apps, rewrite the results viewer with streamlit-aggrid

**Status: implemented.** Everything below was built as described, with one
open risk still unverified (the `streamlit-aggrid` version bound — see the
end of this document and the "Not yet verified" note at the very bottom).

## What you asked for

1. `Probabilistic_Evaluation.ui` should contain exactly two files: one for
   the clinical goals editor, one for the results viewer.
2. Both should use Streamlit (today only the editor does; the viewer is a
   plain `http.server` + hand-written HTML/JS page).
3. The results viewer's HTML/JS is replaced by a native Streamlit workflow,
   using `streamlit-aggrid` for the tables so the result display keeps its
   current look (draggable rows, colored cells, pass/fail highlighting,
   save/load row order) — added to the `ui` extra's dependencies.
4. A console-script entry point for the results viewer, alongside the
   existing `rt-eval-goals`.

## 1. New file layout

| Today | After |
|---|---|
| `src/Probabilistic_Evaluation/io/clinicalGoalsEditor.py` | `src/Probabilistic_Evaluation/ui/clinicalGoalsEditor.py` (moved; its page/widgets are untouched — no AgGrid, same layout — only its launcher plumbing is extracted into the shared helper, see §3) |
| `src/Probabilistic_Evaluation/ui/JSONreader.py` | `src/Probabilistic_Evaluation/ui/resultsViewer.py` (rewritten, see §2) |
| *(new)* | `src/Probabilistic_Evaluation/ui/_launch.py` — shared subprocess launcher used by both (see §3) |

`Probabilistic_Evaluation.ui` becomes the single home for both optional,
Streamlit-based tools; `Probabilistic_Evaluation.io` goes back to being just
`dicomIO` / `clinicalgoalsIO`.

✅ **Resolved: clean move, no shim.** Moving `clinicalGoalsEditor` out of
`io` means `Probabilistic_Evaluation.io.edit_clinical_goals` (added last
turn) stops existing — call sites switch to
`Probabilistic_Evaluation.ui.edit_clinical_goals`. Since this was only
introduced one commit ago and the package isn't published yet, this is a
clean move with no backward-compatible re-export left behind in `io`. The
editor's own contents are untouched — only its location changes.

Updated `ui/__init__.py`:

```python
"""Optional interactive tooling, installed with the ``ui`` extra
(``pip install rt-probabilistic-evaluation[ui]``).

Importing this subpackage never requires the extra's dependencies to be
installed; only actually launching a tool does (each entry point checks
for ``streamlit`` and raises a clear error if it's missing).
"""

from . import clinicalGoalsEditor
from . import resultsViewer
from .clinicalGoalsEditor import edit_clinical_goals
from .resultsViewer import view_results

__all__ = ["clinicalGoalsEditor", "resultsViewer", "edit_clinical_goals", "view_results"]
```

`io/__init__.py` drops the `clinicalGoalsEditor` import/export it currently has,
back to just `dicomIO` (+ the pre-existing `clinicalgoalsIO` entry in `__all__`,
which I noticed is not actually imported there — a pre-existing inconsistency,
left untouched since it's unrelated to this change).

## 2. Rewriting the results viewer as a Streamlit app

### 2.1 What stays as-is

`load_results(folder)` and `build_payload(results)` are pure, framework-free
functions (they already don't touch the HTTP server or the HTML) — they move
into `resultsViewer.py` unchanged.

### 2.2 What's new: `_with_metrics`, the ported per-row calculation

The old page computed three numbers **client-side, in the browser**, and
recomputed them on every drag-reorder — this is why row order matters:

- `prg` — weighted count of scenarios where *this one* goal passes.
- `pr` = `prg / probability mass` — that goal's own pass probability.
- `cpr` — the *joint* probability that this goal **and every goal above it
  in the current row order** all pass (a running logical AND over
  `success`, weighted and normalized).

This becomes a pure Python function so it's unit-testable (it never was
before — it only existed as inline JS):

```python
def _with_metrics(rows: list[dict], weights: list[float], prob_mass: float) -> list[dict]:
    """Attach pr/prg/cpr to each row; order-dependent, like the original JS `compute()`."""
    running = [True] * len(weights)
    out = []
    for r in rows:
        prg = sum(w for w, ok in zip(weights, r["success"]) if ok)
        pr = prg / prob_mass if prob_mass else 0.0
        running = [keep and ok for keep, ok in zip(running, r["success"])]
        cpr = (sum(w for w, keep in zip(weights, running) if keep) / prob_mass) if prob_mass else 0.0
        out.append({**r, "pr": pr, "prg": prg, "cpr": cpr})
    return out
```

I'll add a real test for this (`tests/test_ui/test_results_viewer_metrics.py`)
exercising a couple of known success-array/weight combinations — this logic
had zero test coverage before.

### 2.3 The Streamlit page (`_run_app`) — one page per result file

✅ **Resolved: one browser tab per result JSON file**, matching today's
exact behavior rather than stacking every file into one page. Concretely,
`_run_app` renders a *single* dataset (one file), the same way
`clinicalGoalsEditor._run_app` renders a single clinical-goals file — it's
`view_results()`/the launcher (§3) that fans out to one `streamlit run`
subprocess per `*.json` file found in the folder, each on its own port with
its own browser tab, closable independently:

```python
def _load_one(json_path: Path) -> dict:
    """Build the single-dataset payload for one result JSON file (reuses build_payload)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return build_payload({json_path.stem: {"data": data, "path": json_path}})[0]


def _run_app(json_path: Path, state_file: Path | None) -> None:
    import streamlit as st
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

    st.set_page_config(page_title=f"Evaluation Viewer - {json_path.name}", page_icon="📊", layout="wide")
    ds = _load_one(json_path)

    st.title(f"{ds['key']} — {ds['probability'] * 100:.1f}%")
    _render_probabilistic_table(ds)   # AgGrid, draggable, colored, Save/Load order
    _render_nominal_table(ds)         # AgGrid, read-only, pass/fail highlighting
    _start_liveness_thread(state_file)  # same heartbeat pattern as the editor, so the
                                         # launcher can tell when this tab was closed
```

`load_results(folder)` (the folder-wide scan, with its parse-error
printing) is no longer used by the page itself — each page now loads just
its own file via `_load_one`. `view_results()` in §3 finds which files to
launch with a plain `sorted(folder.glob("*.json"))`, not `load_results`,
since it only needs filenames, not parsed content.

`_render_probabilistic_table`:
- Row order for dataset `ds["key"]` is tracked in
  `st.session_state["row_order::" + ds["key"]]` as a list of
  `f"{roi_name}||{clinical_goal}"` keys (same key format the old
  `order.txt` files already use, so any existing saved order file stays
  compatible) — initialized from `ds["prob_rows"]`'s natural order the first
  time a dataset is seen.
- Every rerun: rebuild the row list in that stored order, run it through
  `_with_metrics`, hand the resulting `pandas.DataFrame` to `AgGrid(...)`
  with `rowDragManaged=True` (a drag handle on the first column, matching
  the old "⋮" handle) and `update_mode=GridUpdateMode.MODEL_CHANGED`.
- If the grid reports a different row order back (the user dragged a row),
  save that order into session state and `st.rerun()` — the next run
  recomputes `pr`/`prg`/`cpr` for the *new* order and redraws with updated
  colors, exactly like the old client-side recompute-on-drop.
- Column coloring is ported **verbatim** as `JsCode` (the exact same
  red→yellow→green gradient function `heat()` and the pass/fail text
  styling), passed as each numeric column's `cellStyle` — ag-Grid renders
  it client-side just like the hand-written HTML did, so no behavior is
  reinterpreted, only relocated.
- "💾 Save order" / "📂 Load order" buttons keep writing/reading the sibling
  `order.txt` file next to the source JSON, in the same one-key-per-line
  format as today.
- Empty state: `st.caption("No probabilistic objectives in this file.")`
  instead of an empty grid, matching the old `.empty-note`.

`_render_nominal_table` is the same idea without drag support: since AgGrid
is confirmed for the viewer (not the editor), it's used for both the
probabilistic *and* the nominal table, reusing the same pass/fail `JsCode`
styling for visual consistency — the nominal table just has row-dragging
disabled.

## 3. Launcher and entry point

### 3.1 Shared helper: `Probabilistic_Evaluation.ui._launch`

`edit_clinical_goals`'s current body (below its file-exists/streamlit
checks) — build a temp state file, spawn
`streamlit run <script> --server.port ... -- <args>`, poll the state file
for a heartbeat until it's closed/stale/never-connected, clean up the
process — is lifted out into one reusable function:

```python
# Probabilistic_Evaluation/ui/_launch.py
def run_streamlit_script(script_path: Path, args: list[str]) -> dict:
    """Run one Streamlit page as a subprocess until its browser tab is closed.

    Blocks until the page's liveness heartbeat goes stale, the page reports
    itself closed, or no browser ever connects (raises RuntimeError in that
    last case). Returns the final state dict written by the page (e.g.
    ``{"saved": True, "closed": True}``), so callers can inspect whatever
    fields their page writes.
    """
```

This is a mechanical extraction — same subprocess arguments, same polling
loop, same timeouts (`_STALE_SECONDS`, `_BROWSER_GRACE_SECONDS`) — just
moved so both apps share it instead of duplicating it. The heartbeat
writer (`_start_liveness_thread` / `_write_state`) also moves here since
both pages need it identically; `clinicalGoalsEditor.py` and
`resultsViewer.py` each just call it from inside their own `_run_app`.

`edit_clinical_goals` becomes a thin wrapper:

```python
def edit_clinical_goals(json_path: str | Path) -> bool:
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Clinical goals file not found: {path}")
    _require_ui_extra()  # was the inline find_spec("streamlit") check
    state = run_streamlit_script(Path(__file__), [str(path.resolve())])
    return bool(state.get("saved", False))
```

### 3.2 `view_results`: one subprocess per result file

Per the "one page per file" decision (§2.3), `view_results` fans out —
launches one `run_streamlit_script` call per `*.json` file, each in its own
thread (so they run concurrently, one browser tab each), and returns once
every tab has been closed:

```python
def view_results(folder: str | Path) -> None:
    """Open one results-viewer tab per ``*.json`` file found in ``folder``."""
    path = Path(folder)
    if not path.is_dir():
        raise FileNotFoundError(f"Results folder not found: {path}")
    _require_ui_extra()
    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.json result files found in {path}")

    threads = [
        threading.Thread(target=run_streamlit_script, args=(Path(__file__), [str(f.resolve())]))
        for f in json_files
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # returns once every opened tab has been closed
```

```python
def _cli(prog_name: str = "rt-view-results") -> None:
    args = sys.argv[1:]
    folder = args[0] if args else "Results"   # same default the old __main__ hardcoded
    if len(args) > 1:
        sys.exit(f"usage: {prog_name} [folder]")
    view_results(folder)
```

New console script in `pyproject.toml`:

```toml
[project.scripts]
rt-eval-goals   = "Probabilistic_Evaluation.ui.clinicalGoalsEditor:_cli"   # path updated (moved from io)
rt-view-results = "Probabilistic_Evaluation.ui.resultsViewer:_cli"
```

✅ **Resolved: extract a shared launcher instead of duplicating.**
`edit_clinical_goals` and the new `view_results` would otherwise duplicate
~90 lines of identical subprocess/temp-dir/heartbeat-polling code. That
logic moves into a small private helper,
`Probabilistic_Evaluation.ui._launch.run_streamlit_script(script_path, args) -> dict`
(returns the raw state dict the page wrote — `edit_clinical_goals` reads
its `"saved"` key, `view_results` doesn't need any key from it), used by
both `_cli`s. This does mean
`clinicalGoalsEditor.py`'s current inline launcher code (`_free_port`,
the whole body of `edit_clinical_goals` below the file-existence/streamlit
checks) moves into `_launch.py` and `edit_clinical_goals` becomes a thin
wrapper around it — a refactor of already-working code, but a mechanical
one (same subprocess arguments, same polling loop, same timeouts).

## 4. Dependencies

Add `streamlit-aggrid` to the `ui` extra:

```toml
ui = [
  "streamlit>=1.42,<2.0",
  "streamlit-aggrid>=1.0,<2.0"
]
```

Import stays lazy (inside `_run_app`, never at module top level) — same
rule the whole `ui` package already follows, so `pip install
rt-probabilistic-evaluation` (no extra) still imports
`Probabilistic_Evaluation.ui.resultsViewer` fine, and only *running* it
without the extra raises the usual clear `RuntimeError`.

**⚠ Flag, not really a confirm:** `streamlit-aggrid` has a history of
lagging behind new Streamlit releases (it wraps a specific ag-Grid JS build
and its own component protocol version). `>=1.0,<2.0` is a starting point,
not verified — I'll need to actually install both packages together and
smoke-test the grid once implementing, and may need to narrow the range if
the current `streamlit>=1.42` floor turns out incompatible with whatever
`streamlit-aggrid` release supports it.

## 5. Other files touched

- `tests/test_utils/test_jsonreader_import.py` → renamed to
  `tests/test_ui/test_results_viewer_import.py`, importing
  `Probabilistic_Evaluation.ui.resultsViewer` instead of
  `Probabilistic_Evaluation.ui.JSONreader`.
- New `tests/test_ui/test_results_viewer_metrics.py` for `_with_metrics`
  (see §2.2).
- `README.md`: update the `edit_clinical_goals` import path
  (`Probabilistic_Evaluation.ui`, not `.io`) and add a matching "results
  viewer" subsection with the `rt-view-results` command and a
  `view_results("Results")` Python example.
- `docs/proposals/streamlit-ui-extra.md`: update the paths it references
  (`Probabilistic_Evaluation.io.clinicalGoalsEditor` → `...ui.clinicalGoalsEditor`)
  so it stays accurate as a record rather than describing a layout that no
  longer exists.
- CI (`.github/workflows/testing_pipeline.yml`): no structural change needed
  — the existing `["dev", "dev,ui"]` matrix already exercises both "extra
  absent" (catches an accidental eager `streamlit`/`st_aggrid` import) and
  "extra present" paths.

## Summary

All decisions resolved:

1. ✅ Clean move of `clinicalGoalsEditor` into `ui`, no compatibility shim left in `io`.
2. ✅ Editor and viewer are two separate Streamlit apps/pages, never combined.
3. ✅ AgGrid used only in the results viewer; the editor's existing plain-widget page is untouched.
4. ✅ Shared `_launch.py` helper extracted, used by both `_cli`s instead of duplicated subprocess code.
5. ✅ One results-viewer browser tab per result JSON file (`view_results` fans out one subprocess per file), matching today's exact behavior rather than stacking files onto one page.

Remaining risk, not a question — flagged so it isn't a surprise later:

6. `streamlit-aggrid>=1.0,<2.0` is an unverified starting version bound; needs pinning precisely once the packages are actually installed together and the grid is smoke-tested against the `streamlit>=1.42` floor. **Not yet verified** — this environment has no working Python interpreter to install into and run (see Implementation notes below).

## Implementation notes (found while building, not anticipated in the plan above)

- `from . import _launch` (a relative import) breaks when either page is run
  standalone via `streamlit run clinicalGoalsEditor.py` / `streamlit run
  resultsViewer.py` directly, since Streamlit executes the file with no
  package context. Both files now import `_launch` through a small
  `_import_launch()` helper that tries the normal relative import first and
  falls back to loading `_launch.py` by file path
  (`importlib.util.spec_from_file_location`) — so both the installed-package
  path and the standalone-script path work.
- **Not run.** This sandbox has no working Python interpreter (only a
  non-functional `python.exe` alias), so none of this — including the new
  `tests/test_ui/` tests, the AgGrid rendering code, or the `_launch.py`
  extraction — has actually been executed. Please run
  `pip install -e .[dev,ui]` and `pytest` before relying on this, and smoke-test
  both `rt-eval-goals` and `rt-view-results` against a real result folder.
