"""Web based (Streamlit) viewer for probabilistic evaluation result JSON files.

Each result JSON file is expected to have a ``"Table"`` entry (a list of per-ROI
clinical-goal rows) and, for rows evaluated across scenarios, a ``"Probability
Array"`` of per-scenario weights and a ``"Probability Mass"``. Call
:func:`view_results` with a folder: it opens one browser tab per ``*.json``
file found there and returns once every tab has been closed.

This module only depends on the standard library, Streamlit, pandas (a core
dependency of the package, so always available), streamlit-aggrid, and its
sibling :mod:`_launch` module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pandas as pd


def _import_launch():
    """Import the sibling ``_launch`` module.

    A plain ``from . import _launch`` breaks when this file is executed
    standalone (``streamlit run resultsViewer.py``), since that runs it with
    no package context and relative imports raise ``ImportError``. Fall back
    to loading it directly by file path in that case.
    """
    try:
        from . import _launch as module
    except ImportError:
        spec = importlib.util.spec_from_file_location("_launch", Path(__file__).resolve().parent / "_launch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


_launch = _import_launch()
logger = _launch.logger


# ---------------------------
# Load JSON files
# ---------------------------
def _is_order_file(path: Path) -> bool:
    """True if ``path`` looks like a saved row-order file (e.g. ``foo_order.txt``), not a result file."""
    return "order" in path.stem.lower()


def load_results(folder: Path) -> dict:
    """Read every ``*.json`` file in ``folder`` into ``{stem: {"data": ..., "path": ...}}``.

    Files whose name contains ``order`` (e.g. a saved ``foo_order.txt`` that ended up with a
    ``.json`` extension) are skipped, since they hold a saved row order, not result data.
    """
    results = {}
    print(f"\n=== Loading folder: {folder} ===")

    for f in Path(folder).glob("*.json"):
        if _is_order_file(f):
            print(f"Skipping order file: {f.name}")
            continue
        print(f"Loading: {f.name} ... ", end="")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results[f.stem] = {"data": data, "path": f}
            print(f"✅ Loaded: {f.name}")
        except Exception as e:
            print(f"❌ Failed: {f.name} -> {e}")
            logger.debug(f"Could not load the result file {f}.", exc_info=True)

    return results


# ---------------------------
# Convert JSON -> payload
# ---------------------------
def build_payload(results: dict) -> list[dict]:
    """Turn ``load_results``' output into one display-ready dataset dict per file."""
    payload = []

    def format_nominal(goal: str, raw_value):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return str(raw_value)

        if goal.startswith("V"):
            if goal.endswith("cc"):
                return "{:.2f}cc".format(value)
            return "{:.2f}%".format(value*100)
        return "{:.2f}Gy".format(value)

    for key, obj in results.items():
        data = obj["data"]
        path = obj["path"]

        probs = data.get("Probability Array", [])
        table = data.get("Table", [])

        if not table:
            continue

        prob_rows = []
        nominal_rows = []

        for row in table:
            goal = row.get("Clinical Goal", "")
            nominal = format_nominal(goal, row.get("Nominal Value", "N/A"))

            is_probabilistic = bool(row.get("Probabilistic Objective", False))
            success = row.get("Success Array", [])

            # Backward compatibility if the explicit flag is absent.
            if not is_probabilistic and isinstance(success, list) and probs:
                is_probabilistic = len(success) == len(probs)

            if is_probabilistic and isinstance(success, list) and probs and len(success) == len(probs):
                prob_rows.append(
                    {
                        "roi_name": row.get("Mask Name", ""),
                        "clinical_goal": goal,
                        "nominal": nominal,
                        "nominal_passed": row.get("Nominal Success"),
                        "success": success,
                    }
                )
            else:
                nominal_rows.append(
                    {
                        "roi_name": row.get("Mask Name", ""),
                        "clinical_goal": goal,
                        "nominal": nominal,
                        "nominal_passed": row.get("Nominal Success"),
                    }
                )

        payload.append(
            {
                "key": key,
                "file_path": str(path),
                "probability": float(data.get("Probability Mass", sum(probs) if probs else 1.0)),
                "prob_rows": prob_rows,
                "nominal_rows": nominal_rows,
                "weights": probs,
            }
        )

    return payload


def _load_one(json_path: Path) -> dict:
    """Build the single-dataset payload for one result JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return build_payload({json_path.stem: {"data": data, "path": json_path}})[0]


# ---------------------------
# Per-row metrics (ported from the old client-side JS `compute()`)
# ---------------------------
def _row_key(row: dict) -> str:
    """Same ``roi||goal`` key format the old page's ``order.txt`` files already use."""
    return f"{row['roi_name']}||{row['clinical_goal']}"


def _with_metrics(rows: list[dict], weights: list[float], prob_mass: float) -> list[dict]:
    """Attach pr/prg/cpr to each row; order-dependent, like the original JS ``compute()``.

    - ``prg``: weighted count of scenarios where this one goal passes.
    - ``pr`` = ``prg / prob_mass``: that goal's own pass probability.
    - ``cpr``: the joint probability that this goal *and every goal above it in
      ``rows``' current order* all pass (a running weighted logical AND).
    """
    running = [True] * len(weights)
    out = []
    for r in rows:
        prg = sum(w for w, ok in zip(weights, r["success"]) if ok)
        pr = prg / prob_mass if prob_mass else 0.0
        running = [keep and ok for keep, ok in zip(running, r["success"])]
        cpr = (sum(w for w, keep in zip(weights, running) if keep) / prob_mass) if prob_mass else 0.0
        out.append({**r, "pr": pr, "prg": prg, "cpr": cpr})
    return out


# ---------------------------
# Streamlit web app  (executed by ``streamlit run resultsViewer.py -- <result.json> <state>``)
# ---------------------------

_HEAT_CELL_STYLE = """
function(params) {
    if (params.value === null || params.value === undefined) { return {}; }
    let v = Math.max(0, Math.min(1, params.value));
    let r, g;
    if (v < 0.5) { r = 255; g = Math.round(255 * (v / 0.5)); }
    else { g = 255; r = Math.round(255 * (1 - (v - 0.5) / 0.5)); }
    return {backgroundColor: `rgba(${r}, ${g}, 0, 0.35)`, textAlign: 'right'};
}
"""

_PASS_FAIL_CELL_STYLE = """
function(params) {
    if (params.data.nominal_passed === true) { return {color: '#1b8a3e', fontWeight: '600'}; }
    if (params.data.nominal_passed === false) { return {color: '#c62828', fontWeight: '600'}; }
    return {};
}
"""

_NUMBER_FORMATTER = "function(params) { return params.value === null || params.value === undefined ? '' : params.value.toFixed(3); }"

# A table with more rows than this gets a fixed height and scrolls inside the grid, instead of pushing the rest of the
# page down. Row and header heights are those of the default streamlit-aggrid theme; if they ever change, the grid still
# scrolls correctly and only the last visible row is cut differently.
_MAX_VISIBLE_ROWS = 15
_ROW_HEIGHT = 29
_HEADER_HEIGHT = 33


def _grid_height(n_rows: int) -> int | None:
    """Grid height in px for ``n_rows`` rows: None (fit every row) up to ``_MAX_VISIBLE_ROWS``, a fixed height above."""
    if n_rows <= _MAX_VISIBLE_ROWS:
        return None
    return _HEADER_HEIGHT + _MAX_VISIBLE_ROWS * _ROW_HEIGHT + 2  # + the 1 px top and bottom border of the grid


def _render_probabilistic_table(ds: dict) -> list[dict]:
    """Draggable, colored table of probabilistic objectives, with Save/Load order.

    Returns the rows as currently displayed (in the user's chosen order, with the
    pr/prg/cpr metrics computed for that order) so callers can reuse them (e.g. to
    export the results).
    """
    import streamlit as st
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, JsCode

    if not ds["prob_rows"]:
        st.caption("No probabilistic objectives in this file.")
        return []

    order_key = f"row_order::{ds['key']}"
    if order_key not in st.session_state:
        st.session_state[order_key] = [_row_key(r) for r in ds["prob_rows"]]

    # Bumped whenever the order is set programmatically (e.g. by "Load order") so the grid
    # below gets a fresh `key` and is fully remounted. Once the user has dragged a row, the
    # AgGrid component keeps managing row order client-side and ignores new server-provided
    # data with the same content (order is not part of its change-detection hash), so simply
    # re-passing a reordered dataframe under the same key would otherwise be a no-op.
    version_key = f"row_order_version::{ds['key']}"
    st.session_state.setdefault(version_key, 0)

    rows_by_key = {_row_key(r): r for r in ds["prob_rows"]}
    ordered_rows = [rows_by_key[k] for k in st.session_state[order_key] if k in rows_by_key]
    ordered_rows += [r for r in ds["prob_rows"] if _row_key(r) not in st.session_state[order_key]]

    computed_rows = _with_metrics(ordered_rows, ds["weights"], ds["probability"])
    df = pd.DataFrame(
        [
            {
                "_drag": "",
                "roi_name": r["roi_name"],
                "clinical_goal": r["clinical_goal"],
                "nominal": r["nominal"],
                "nominal_passed": r["nominal_passed"],
                "pr": r["pr"],
                "prg": r["prg"],
                "cpr": r["cpr"],
            }
            for r in computed_rows
        ]
    )

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=False, filter=False, editable=False)
    gb.configure_column("_drag", header_name="", width=40, rowDrag=True, suppressMenu=True)
    gb.configure_column("roi_name", header_name="ROI")
    gb.configure_column("clinical_goal", header_name="Clinical Goal")
    gb.configure_column("nominal", header_name="Nominal", cellStyle=JsCode(_PASS_FAIL_CELL_STYLE))
    gb.configure_column("nominal_passed", hide=True)
    for field, header in (("pr", "Passing"), ("prg", "Global"), ("cpr", "Cumulative")):
        gb.configure_column(
            field,
            header_name=header,
            type=["numericColumn"],
            valueFormatter=JsCode(_NUMBER_FORMATTER),
            cellStyle=JsCode(_HEAT_CELL_STYLE),
        )
    gb.configure_grid_options(rowDragManaged=True, animateRows=True)

    response = AgGrid(
        df,
        gridOptions=gb.build(),
        height=_grid_height(len(df)),  # dragging a row to the top or bottom edge scrolls the grid
        update_on=["rowDragEnd"],
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        key=f"prob_grid_{ds['key']}_v{st.session_state[version_key]}",
    )

    new_order = [_row_key(r) for r in response["data"].to_dict("records")]
    if new_order != st.session_state[order_key]:
        st.session_state[order_key] = new_order
        st.rerun()  # recompute pr/prg/cpr for the new order and redraw with updated colors

    # Named after the result file itself (e.g. ``demo_results_order.txt``) so several result
    # files sharing the same folder don't clobber each other's saved order.
    order_file = Path(ds["file_path"]).with_name(f"{ds['key']}_order.txt")
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("💾 Save order", key=f"save_order_{ds['key']}"):
        order_file.write_text("\n".join(st.session_state[order_key]), encoding="utf-8")
        st.toast(f"Saved order to {order_file.name}")
    if c2.button("📂 Load order", key=f"load_order_{ds['key']}"):
        if order_file.exists():
            loaded = [k for k in order_file.read_text(encoding="utf-8").splitlines() if k]
            if loaded:
                st.session_state[order_key] = loaded
                st.session_state[version_key] += 1
                st.rerun()
            else:
                st.warning(f"{order_file.name} is empty")
        else:
            st.warning(f"No saved order ({order_file.name} not found)")

    return computed_rows


def _render_nominal_table(ds: dict) -> None:
    """Read-only table of non-probabilistic objectives, with the same pass/fail styling."""
    import streamlit as st
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

    if not ds["nominal_rows"]:
        st.caption("No non-probabilistic objectives in this file.")
        return

    df = pd.DataFrame(
        [
            {
                "roi_name": r["roi_name"],
                "clinical_goal": r["clinical_goal"],
                "nominal": r["nominal"],
                "nominal_passed": r["nominal_passed"],
            }
            for r in ds["nominal_rows"]
        ]
    )

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=False, filter=False, editable=False)
    gb.configure_column("roi_name", header_name="ROI")
    gb.configure_column("clinical_goal", header_name="Clinical Goal")
    gb.configure_column("nominal", header_name="Nominal", cellStyle=JsCode(_PASS_FAIL_CELL_STYLE))
    gb.configure_column("nominal_passed", hide=True)

    AgGrid(
        df,
        gridOptions=gb.build(),
        height=_grid_height(len(df)),
        update_on=[],
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        key=f"nominal_grid_{ds['key']}",
    )


def _default_export_name(json_path: Path) -> str:
    """Default "Save results" filename: derived from the displayed file's name, but never
    equal to it, so saving never silently overwrites the source result file."""
    return f"{json_path.stem}_export.json"


def _save_results(json_path: Path, filename: str, order_keys: list[str]) -> Path:
    """Save a copy of the source result JSON next to ``json_path``, in the exact same
    schema (``Table``/``Probability Array``/``Probability Mass``/...), so it can be
    reopened later exactly like any other result file.

    Only the ``Table`` rows' order is changed, to match ``order_keys`` (the currently
    displayed ``roi||goal`` order of the probabilistic objectives) - every field, of
    every row (including ones not covered by ``order_keys``, e.g. non-probabilistic
    objectives), is copied through unchanged.

    Raises ``ValueError`` if the target name resolves to ``json_path`` itself, to avoid
    silently overwriting the source result file.
    """
    name = filename.strip() or _default_export_name(json_path)
    if not name.lower().endswith(".json"):
        name += ".json"
    export_path = json_path.with_name(name)
    if export_path.resolve() == json_path.resolve():
        raise ValueError(f"{name!r} is the file currently being displayed - choose a different name")

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    table = raw.get("Table", [])

    def _table_row_key(row: dict) -> str:
        return f"{row.get('Mask Name', '')}||{row.get('Clinical Goal', '')}"

    by_key = {_table_row_key(row): row for row in table}
    ordered = [by_key[k] for k in order_keys if k in by_key]
    ordered += [row for row in table if _table_row_key(row) not in order_keys]
    raw["Table"] = ordered

    export_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return export_path


def _run_app(json_path: Path, state_file: Path | None) -> None:
    """Body of the Streamlit page: renders the single dataset for ``json_path``."""
    import streamlit as st

    st.set_page_config(page_title=f"Evaluation Viewer - {json_path.name}", page_icon="📊", layout="wide")
    _launch.start_liveness_thread(state_file)

    try:
        ds = _load_one(json_path)
    except (OSError, ValueError) as e:
        st.error(f"Cannot read {json_path}:\n\n{e}")
        st.stop()

    st.title(f"{ds['key']} — {ds['probability'] * 100:.1f}%")
    st.caption(f"File: `{json_path}`")

    # Rendered here (top of the page) but filled in further down, once the current display
    # order/computed metrics are known - a Streamlit container keeps its declared position
    # while letting later code write into it.
    st.subheader("Save results")
    save_container = st.container()

    st.subheader("Probabilistic objectives")
    prob_computed_rows = _render_probabilistic_table(ds)

    st.subheader("Non-probabilistic objectives")
    _render_nominal_table(ds)

    c1, c2 = save_container.columns([4, 1])
    export_name = c1.text_input(
        "File name",
        value=_default_export_name(json_path),
        key=f"export_name_{ds['key']}",
        label_visibility="collapsed",
    )
    if c2.button("💾 Save results", key=f"save_results_{ds['key']}"):
        try:
            order_keys = [_row_key(r) for r in prob_computed_rows]
            saved_path = _save_results(json_path, export_name, order_keys)
        except ValueError as e:
            st.warning(str(e))
        else:
            st.toast(f"Saved results to {saved_path.name}")


# ----------------------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------------------


def view_results(folder: str | Path) -> None:
    """Open one results-viewer browser tab per result JSON file.

    Each file gets its own local Streamlit server and its own browser tab,
    exactly like each clinical goals file gets its own editor session. The
    call blocks until every opened tab has been closed.

    Parameters
    ----------
    folder : str or Path
        Either a folder containing one or more probabilistic evaluation result JSON
        files (a tab is opened for every ``*.json`` file found there - the default,
        e.g. what ``rt-view-results`` uses to show everything that's been saved so
        far), or the path to a single result JSON file (only that one file is shown,
        e.g. what ``evaluator.py`` uses right after computing it).
    """
    path = Path(folder)
    if path.is_file():
        json_files = [path]
    elif path.is_dir():
        json_files = sorted(f for f in path.glob("*.json") if not _is_order_file(f))
        if not json_files:
            raise FileNotFoundError(f"No *.json result files found in {path}")
    else:
        raise FileNotFoundError(f"Results file or folder not found: {path}")
    _launch.require_streamlit("The results viewer")

    logger.debug(f"Opening the results viewer for {len(json_files)} result files in {path}: {[f.name for f in json_files]}")
    errors: list[Exception] = []

    def _run(f: Path) -> None:
        try:
            _launch.run_streamlit_script(Path(__file__), [str(f.resolve())])
        except Exception as e:  # surfaced to the caller once every tab has been dealt with
            logger.debug(f"The results viewer for {f.name} failed.", exc_info=True)
            errors.append(e)

    threads = [threading.Thread(target=_run, args=(f,)) for f in json_files]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("; ".join(str(e) for e in errors))


def _inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


def _cli(prog_name: str = "rt-view-results") -> None:
    """Command-line entry point: opens the viewer for every result file in a folder.

    Installed as the ``rt-view-results`` console script (available once the
    ``ui`` extra is installed) and also used for a plain
    ``python resultsViewer.py [folder]`` invocation. Pass a single ``*.json`` file
    instead of a folder to only view that one result.
    """
    args = sys.argv[1:]
    if len(args) > 1:
        sys.exit(f"usage: {prog_name} [folder_or_file]")
    folder = args[0] if args else "Results"  # same default the original script hardcoded
    view_results(folder)


if __name__ == "__main__":
    args = sys.argv[1:]
    if _inside_streamlit():  # started through `streamlit run`
        if not args:
            sys.exit("usage: streamlit run resultsViewer.py -- <result.json> [<state file>]")
        _run_app(Path(args[0]), Path(args[1]) if len(args) > 1 else None)
    else:  # started with plain python (or the rt-view-results console script)
        _cli(f"python {Path(__file__).name}")
