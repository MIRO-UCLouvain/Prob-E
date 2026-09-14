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


# ---------------------------
# Load JSON files
# ---------------------------
def load_results(folder: Path) -> dict:
    """Read every ``*.json`` file in ``folder`` into ``{stem: {"data": ..., "path": ...}}``."""
    results = {}
    print(f"\n=== Loading folder: {folder} ===")

    for f in Path(folder).glob("*.json"):
        print(f"Loading: {f.name} ... ", end="")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results[f.stem] = {"data": data, "path": f}
            print(f"✅ Loaded: {f.name}")
        except Exception as e:
            print(f"❌ Failed: {f.name} -> {e}")

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
            return str(round(value * 100, 2)) + "%"
        return str(round(value, 2)) + "Gy"

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


def _render_probabilistic_table(ds: dict) -> None:
    """Draggable, colored table of probabilistic objectives, with Save/Load order."""
    import streamlit as st
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode

    if not ds["prob_rows"]:
        st.caption("No probabilistic objectives in this file.")
        return

    order_key = f"row_order::{ds['key']}"
    if order_key not in st.session_state:
        st.session_state[order_key] = [_row_key(r) for r in ds["prob_rows"]]

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
    gb.configure_grid_options(rowDragManaged=True, animateRows=True, domLayout="autoHeight")

    response = AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        key=f"prob_grid_{ds['key']}",
    )

    new_order = [_row_key(r) for r in response["data"].to_dict("records")]
    if new_order != st.session_state[order_key]:
        st.session_state[order_key] = new_order
        st.rerun()  # recompute pr/prg/cpr for the new order and redraw with updated colors

    order_file = Path(ds["file_path"]).parent / "order.txt"
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("💾 Save order", key=f"save_order_{ds['key']}"):
        order_file.write_text("\n".join(st.session_state[order_key]), encoding="utf-8")
        st.toast(f"Saved order to {order_file.name}")
    if c2.button("📂 Load order", key=f"load_order_{ds['key']}"):
        if order_file.exists():
            loaded = [k for k in order_file.read_text(encoding="utf-8").splitlines() if k]
            st.session_state[order_key] = loaded
            st.rerun()
        else:
            st.warning("No saved order")


def _render_nominal_table(ds: dict) -> None:
    """Read-only table of non-probabilistic objectives, with the same pass/fail styling."""
    import streamlit as st
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

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
    gb.configure_grid_options(domLayout="autoHeight")

    AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        key=f"nominal_grid_{ds['key']}",
    )


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

    st.subheader("Probabilistic objectives")
    _render_probabilistic_table(ds)

    st.subheader("Non-probabilistic objectives")
    _render_nominal_table(ds)


# ----------------------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------------------


def view_results(folder: str | Path) -> None:
    """Open one results-viewer browser tab per ``*.json`` file found in ``folder``.

    Each file gets its own local Streamlit server and its own browser tab,
    exactly like each clinical goals file gets its own editor session. The
    call blocks until every opened tab has been closed.

    Parameters
    ----------
    folder : str or Path
        Path to a folder containing one or more probabilistic evaluation
        result JSON files.
    """
    path = Path(folder)
    if not path.is_dir():
        raise FileNotFoundError(f"Results folder not found: {path}")
    _launch.require_streamlit("The results viewer")

    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No *.json result files found in {path}")

    errors: list[Exception] = []

    def _run(f: Path) -> None:
        try:
            _launch.run_streamlit_script(Path(__file__), [str(f.resolve())])
        except Exception as e:  # surfaced to the caller once every tab has been dealt with
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
    ``python resultsViewer.py [folder]`` invocation.
    """
    args = sys.argv[1:]
    if len(args) > 1:
        sys.exit(f"usage: {prog_name} [folder]")
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
