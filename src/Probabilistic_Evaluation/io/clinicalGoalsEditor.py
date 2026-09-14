"""Web based (Streamlit) editor for clinical goal JSON files.

The JSON architecture is a list of goal dictionaries, e.g.::

    { "ROI": "PTV",   "type": "Dx",    "dose": 50.4, "volume": 98,           "lower_is_better": false, "priority": 1, "probabilistic": true }
    { "ROI": "Liver", "type": "Dxcc",  "dose": 21.0, "absolute_volume": 700, "lower_is_better": true,  "priority": 1, "probabilistic": false }
    { "ROI": "Liver", "type": "Dmean", "dose": 15.0,                         "lower_is_better": true,  "priority": 1, "probabilistic": false }

Supported types are ``Dx`` / ``Vx`` (volume in %), ``Dxcc`` / ``Vxcc`` (volume in cc),
``Dmean``, ``Dmax`` and ``Dmin``.  Dose is always in Gy and priority is an integer.
``probabilistic`` marks the goals that are evaluated over the scenarios (passing rates);
it defaults to false.  The order of the goals in the file is the order used for the
cumulative passing rates (within equal priorities), so the editor keeps the file order
and only groups the rows by ROI for display.

Call :func:`edit_clinical_goals` with the path of a JSON file: it starts a local
Streamlit server, opens the editor in the web browser and returns once the user
pressed *Close* (or closed the browser tab).  This module only depends on the
standard library and Streamlit, so the package itself is not imported by the app.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

# ----------------------------------------------------------------------------------
# Goal types
# ----------------------------------------------------------------------------------

# Label shown in the drop-down  ->  base type written to the JSON (without "cc")
TYPE_LABELS: dict[str, str] = {
    "Dose at volume (Dx)": "Dx",
    "Volume at dose (Vx)": "Vx",
    "Mean dose (Dmean)": "Dmean",
    "Max dose (Dmax)": "Dmax",
    "Min dose (Dmin)": "Dmin",
}
_BASE_TO_LABEL: dict[str, str] = {base: label for label, base in TYPE_LABELS.items()}
_VOLUME_TYPES = ("Dx", "Vx")

# Canonical spelling for every accepted (case-insensitive) type string
_CANONICAL_TYPES: dict[str, str] = {t.upper(): t for t in ("Dx", "Vx", "Dxcc", "Vxcc", "Dmean", "Dmax", "Dmin")}

_EDITOR_KEYS = ("ROI", "type", "dose", "volume", "absolute_volume", "lower_is_better", "priority", "probabilistic")

# Heartbeat written by the web app while a browser is connected; the launcher treats the
# app as closed when it goes stale
_HEARTBEAT_SECONDS = 1
_STALE_SECONDS = 15
_BROWSER_GRACE_SECONDS = 180
_LIVENESS_THREAD = "clinical-goals-editor-liveness"


# ----------------------------------------------------------------------------------
# Conversion between JSON dictionaries and the flat editor representation
# ----------------------------------------------------------------------------------


def _goal_from_dict(goal: dict) -> dict:
    """Convert one JSON goal into the editor representation.

    The editor representation has the keys ``ROI`` (str), ``base`` (Dx, Vx, Dmean,
    Dmax or Dmin), ``dose`` (float or None), ``volume`` (float or None), ``in_cc``
    (bool), ``lower_is_better`` (bool), ``priority`` (int), ``probabilistic`` (bool)
    and ``extra`` (any additional keys of the original dictionary, kept untouched).
    """
    raw_type = str(goal.get("type", ""))
    canonical = _CANONICAL_TYPES.get(raw_type.upper())
    if canonical is None:
        raise ValueError(f"Unknown clinical goal type '{raw_type}' for ROI '{goal.get('ROI', '?')}'")

    in_cc = canonical.endswith("cc")
    base = canonical[:-2] if in_cc else canonical

    volume = None
    if base in _VOLUME_TYPES:
        volume = goal.get("absolute_volume") if in_cc else goal.get("volume")
        volume = None if volume is None else float(volume)

    dose = goal.get("dose")
    priority = goal.get("priority", 0)
    return {
        "ROI": str(goal.get("ROI", "")),
        "base": base,
        "dose": None if dose is None else float(dose),
        "volume": volume,
        "in_cc": bool(in_cc),
        "lower_is_better": bool(goal.get("lower_is_better", True)),
        "priority": 0 if priority is None else int(priority),
        "probabilistic": bool(goal.get("probabilistic", False)),
        "extra": {k: v for k, v in goal.items() if k not in _EDITOR_KEYS},
    }


def _goal_to_dict(goal: dict) -> dict:
    """Convert one editor goal back into the JSON architecture.

    Raises ``ValueError`` with a readable message when a field is invalid.
    """
    roi = str(goal["ROI"]).strip()
    if not roi:
        raise ValueError("ROI name is empty")

    if goal["dose"] is None:
        raise ValueError("dose (Gy) is missing")
    dose = float(goal["dose"])

    try:
        priority = int(goal["priority"])
    except (TypeError, ValueError):
        raise ValueError(f"priority '{goal['priority']}' is not an integer") from None

    base = goal["base"]
    out: dict = {"ROI": roi}
    if base in _VOLUME_TYPES:
        if goal["volume"] is None:
            raise ValueError("volume is missing")
        volume = float(goal["volume"])
        if volume.is_integer():
            volume = int(volume)
        if goal["in_cc"]:
            out["type"] = base + "cc"
            out["dose"] = dose
            out["absolute_volume"] = volume
        else:
            if not 0 <= volume <= 100:
                raise ValueError(f"relative volume {volume} % must lie between 0 and 100")
            out["type"] = base
            out["dose"] = dose
            out["volume"] = volume
    else:
        out["type"] = base
        out["dose"] = dose

    out["lower_is_better"] = bool(goal["lower_is_better"])
    out["priority"] = priority
    out["probabilistic"] = bool(goal.get("probabilistic", False))
    out.update(goal.get("extra", {}))
    return out


def load_goals(json_path: str | Path) -> list[dict]:
    """Read a clinical goals JSON file into the editor representation."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("The clinical goals JSON must contain a list of goals")
    return [_goal_from_dict(g) for g in data]


def save_goals(json_path: str | Path, goals: list[dict]) -> None:
    """Write editor goals to ``json_path`` in the one-goal-per-line JSON layout.

    Goals are written in their current (file) order, which is the order used for the
    cumulative passing rates; a blank line separates groups of consecutive goals that
    belong to different ROIs, like the reference files.
    """
    dicts = [_goal_to_dict(g) for g in goals]
    lines = ["["]
    previous_roi = None
    for i, d in enumerate(dicts):
        if previous_roi is not None and d["ROI"] != previous_roi:
            lines.append("")
        comma = "," if i < len(dicts) - 1 else ""
        lines.append("  " + json.dumps(d, ensure_ascii=False) + comma)
        previous_roi = d["ROI"]
    lines.append("]")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _validate(goals: list[dict]) -> list[str]:
    """Return a list of human readable problems, empty when all goals are valid."""
    problems = []
    for i, g in enumerate(goals, start=1):
        try:
            _goal_to_dict(g)
        except ValueError as e:
            problems.append(f"Goal {i} ({g['ROI'] or 'no ROI'}): {e}")
    return problems


def _new_goal() -> dict:
    return {"ROI": "", "base": "Dx", "dose": None, "volume": None, "in_cc": False, "lower_is_better": True, "priority": 0, "probabilistic": False, "extra": {}}


def _snapshot(goals: list[dict]) -> str:
    """Comparable representation of the goals, used to detect unsaved changes."""
    return json.dumps([{k: v for k, v in g.items() if k != "gid"} for g in goals], sort_keys=True, default=str)


# ----------------------------------------------------------------------------------
# Streamlit web app  (executed by ``streamlit run clinicalGoalsEditor.py -- <json> <state>``)
# ----------------------------------------------------------------------------------

_FIELDS = ("ROI", "base", "dose", "volume", "in_cc", "lower_is_better", "priority", "probabilistic")


def _write_state(state_file: Path | None, **updates) -> None:
    """Merge ``updates`` into the small JSON file the launcher polls."""
    if state_file is None:
        return
    try:
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    except (OSError, ValueError):
        state = {}
    state.update(updates)
    state_file.write_text(json.dumps(state), encoding="utf-8")


def _active_sessions() -> int | None:
    """Number of browsers currently connected to this Streamlit server, None if unknown."""
    try:
        from streamlit.runtime import get_instance

        return int(get_instance()._session_mgr.num_active_sessions())
    except Exception:  # private API changed or runtime not started: assume connected
        return None


def _start_liveness_thread(state_file: Path | None) -> None:
    """Write a heartbeat to ``state_file`` while at least one browser is connected.

    This runs server side, so it keeps going while the browser tab is hidden or
    throttled; it only stops when every tab is closed (or the server exits).
    A browser driven timer (``st.fragment(run_every=...)``) is not suitable here
    because browsers slow such timers down drastically for background tabs.
    """
    if state_file is None or any(t.name == _LIVENESS_THREAD for t in threading.enumerate()):
        return

    def loop() -> None:
        while True:
            n = _active_sessions()
            if n is None or n > 0:
                _write_state(state_file, alive=time.time())
            time.sleep(_HEARTBEAT_SECONDS)

    threading.Thread(target=loop, name=_LIVENESS_THREAD, daemon=True).start()


def _sort_goals(goals: list[dict]) -> list[dict]:
    """Display order: stable sort by ROI name (case-insensitive); goals without an ROI go last.

    Only used to lay out the table. The goals list itself keeps the file order, which
    is what the evaluator uses for the cumulative passing rates. Goals of the same ROI
    keep their relative order, so a copy stays next to its source and a goal whose ROI
    is edited is displayed with the group of its new ROI.
    """
    return sorted(goals, key=lambda g: (g["ROI"].strip() == "", g["ROI"].strip().lower()))


def _insert_in_roi_group(goals: list[dict], goal: dict) -> None:
    """Insert ``goal`` after the last goal of the same ROI (file order), or append it."""
    roi = goal["ROI"].strip().lower()
    position = len(goals)
    if roi:
        for i, g in enumerate(goals):
            if g["ROI"].strip().lower() == roi:
                position = i + 1
    goals.insert(position, goal)


def _run_app(json_path: Path, state_file: Path | None) -> None:
    """Body of the Streamlit page."""
    import streamlit as st

    st.set_page_config(page_title=f"Clinical goals - {json_path.name}", page_icon="🎯", layout="wide")
    ss = st.session_state

    # ---------------------------------------------------------- state --
    if ss.get("loaded_path") != str(json_path):
        try:
            goals = load_goals(json_path)
        except (OSError, ValueError) as e:
            st.error(f"Cannot read {json_path}:\n\n{e}")
            st.stop()
        for g in goals:
            g["gid"] = uuid.uuid4().hex
        ss.goals = goals  # file order is kept; the table is only displayed grouped by ROI
        ss.loaded_path = str(json_path)
        ss.saved_snapshot = _snapshot(goals)
        ss.saved = False
        ss.confirm_close = False
        ss.closed = False

    # The goals list in session state is the single source of truth.  Widgets are
    # created with the goal's current value as their default and NEVER seeded
    # through the session-state API: a value seeded that way lingers in Streamlit's
    # internal "old state" and can be returned instead of the widget's real value
    # (observed as a goal "losing" its volume right after saving).  All buttons use
    # callbacks so that no script run is interrupted with st.rerun() before the row
    # widgets have been registered.

    def widget_key(field: str, gid: str) -> str:
        return f"{field}_{gid}"

    def index_of(gid: str) -> int:
        return next(i for i, g in enumerate(ss.goals) if g["gid"] == gid)

    def sync_from_widgets() -> None:
        """Copy the latest widget values into the goals (widget state is updated before the script runs)."""
        for goal in ss.goals:
            for field in _FIELDS:
                key = widget_key(field, goal["gid"])
                if key in ss:
                    goal[field] = TYPE_LABELS[ss[key]] if field == "base" else ss[key]

    sync_from_widgets()

    # ------------------------------------------------------ callbacks --
    # (run by Streamlit before the script body, with the widget values already applied)
    def add_goal() -> None:
        sync_from_widgets()
        goal = _new_goal()
        goal["gid"] = uuid.uuid4().hex
        goal["ROI"] = (ss.get("new_goal_roi") or "").strip()
        same_roi = [g for g in ss.goals if g["ROI"].strip().lower() == goal["ROI"].lower()]
        if same_roi:  # inherit the priority used by the goals of that ROI
            goal["priority"] = same_roi[-1]["priority"]
        _insert_in_roi_group(ss.goals, goal)  # after the last goal of its ROI in the file (or last while the ROI is empty)

    def copy_goal(gid: str) -> None:
        sync_from_widgets()
        i = index_of(gid)
        duplicate = copy.deepcopy(ss.goals[i])
        duplicate["gid"] = uuid.uuid4().hex
        ss.goals.insert(i + 1, duplicate)

    def delete_goal(gid: str) -> None:
        sync_from_widgets()
        del ss.goals[index_of(gid)]

    def save() -> bool:
        sync_from_widgets()
        problems = _validate(ss.goals)
        if problems:
            ss.flash = ("error", "Cannot save, please fix these goals first:\n\n" + "\n".join(f"- {p}" for p in problems))
            return False
        try:
            save_goals(json_path, ss.goals)
        except OSError as e:
            ss.flash = ("error", f"Writing {json_path} failed: {e}")
            return False
        ss.saved_snapshot = _snapshot(ss.goals)
        ss.saved = True
        _write_state(state_file, saved=True)
        ss.flash = ("toast", f"Saved {len(ss.goals)} goals to {json_path.name}")
        return True

    def close() -> None:
        ss.closed = True
        _write_state(state_file, saved=ss.saved, closed=True)

    def on_close() -> None:
        sync_from_widgets()
        if _snapshot(ss.goals) != ss.saved_snapshot:
            ss.confirm_close = True
        else:
            close()

    def on_save_and_close() -> None:
        if save():
            close()

    def on_cancel_close() -> None:
        ss.confirm_close = False

    # ------------------------------------------------------ heartbeat --
    _start_liveness_thread(state_file)

    # --------------------------------------------------------- closed --
    if ss.closed:
        st.success("The editor is closed, you can close this browser tab.")
        st.stop()

    # --------------------------------------------------------- header --
    dirty = _snapshot(ss.goals) != ss.saved_snapshot
    problems = _validate(ss.goals)

    st.title("🎯 Clinical goals")
    st.caption(f"File: `{json_path}`")

    h1, h2, h3, h4 = st.columns([1, 1, 1, 5])
    h1.button("💾 Save", type="primary", disabled=bool(problems), key="save_button", on_click=save)
    h2.button("✖ Close", key="close_button", on_click=on_close)
    if dirty:
        h3.warning("Unsaved changes", icon="⚠️")
    elif ss.saved:
        h3.success("Saved", icon="✅")
    else:
        h3.info("No changes", icon="ℹ️")

    flash = ss.pop("flash", None)
    if flash is not None:
        kind, text = flash
        if kind == "toast":
            st.toast(text, icon="💾")
        else:
            st.error(text)

    if ss.confirm_close:
        with st.container(border=True):
            st.warning("There are unsaved changes. Save them before closing?")
            c1, c2, c3, _ = st.columns([1, 1.3, 1, 5])
            c1.button("Save and close", type="primary", disabled=bool(problems), key="confirm_save_close", on_click=on_save_and_close)
            c2.button("Close without saving", key="confirm_discard", on_click=close)
            c3.button("Cancel", key="confirm_cancel", on_click=on_cancel_close)

    if problems:
        st.error("Some goals are incomplete:\n\n" + "\n".join(f"- {p}" for p in problems))

    # ---------------------------------------------------------- table --
    filter_text = st.text_input("Filter by ROI", value="", placeholder="type part of an ROI name to show only those goals", key="roi_filter").strip().lower()

    widths = [0.4, 2.4, 2.0, 1.1, 1.1, 0.9, 1.1, 0.9, 1.0, 0.45, 0.45]
    header = st.columns(widths, vertical_alignment="bottom")
    for col, label in zip(header, ("#", "ROI", "Type", "Dose (Gy)", "Volume", "in cc", "Lower is better", "Priority", "Probabilistic", "", "")):
        col.markdown(f"**{label}**")

    type_labels = list(TYPE_LABELS)
    # rows are displayed grouped by ROI; "#" is the position in the file (the cumulative evaluation order)
    for goal in _sort_goals(ss.goals):
        if filter_text and filter_text not in goal["ROI"].lower():
            continue
        gid = goal["gid"]
        cols = st.columns(widths, vertical_alignment="center")

        cols[0].markdown(f"{index_of(gid) + 1}")
        goal["ROI"] = cols[1].text_input("ROI", value=goal["ROI"], key=widget_key("ROI", gid), label_visibility="collapsed")
        goal["base"] = TYPE_LABELS[cols[2].selectbox("Type", type_labels, index=type_labels.index(_BASE_TO_LABEL[goal["base"]]), key=widget_key("base", gid), label_visibility="collapsed")]
        goal["dose"] = cols[3].number_input("Dose (Gy)", value=goal["dose"], min_value=0.0, step=0.1, format="%g", placeholder="Gy", key=widget_key("dose", gid), label_visibility="collapsed")

        if goal["base"] in _VOLUME_TYPES:
            goal["in_cc"] = cols[5].checkbox("in cc", value=goal["in_cc"], key=widget_key("in_cc", gid), label_visibility="collapsed")
            # keep label/limits constant: changing them with the unit would reset the widget
            goal["volume"] = cols[4].number_input(
                "Volume", value=goal["volume"], min_value=0.0, step=0.1, format="%g", placeholder="% or cc", key=widget_key("volume", gid), label_visibility="collapsed"
            )
        else:
            cols[4].markdown("<span style='color:grey'>n/a</span>", unsafe_allow_html=True)

        goal["lower_is_better"] = cols[6].checkbox("Lower is better", value=goal["lower_is_better"], key=widget_key("lower_is_better", gid), label_visibility="collapsed")
        goal["priority"] = cols[7].number_input("Priority", value=int(goal["priority"]), min_value=0, step=1, format="%d", key=widget_key("priority", gid), label_visibility="collapsed")
        goal["probabilistic"] = cols[8].checkbox("Probabilistic", value=bool(goal.get("probabilistic", False)), key=widget_key("probabilistic", gid), label_visibility="collapsed", help="Evaluate this goal over the scenarios (passing rate)")

        cols[9].button("", icon=":material/content_copy:", key=f"copy_{gid}", help="Duplicate this goal", on_click=copy_goal, args=(gid,))
        cols[10].button("", icon=":material/delete:", key=f"delete_{gid}", help="Delete this goal", on_click=delete_goal, args=(gid,))

    # ------------------------------------------------------ add goal --
    st.markdown("**Add a goal**")
    a1, a2, _ = st.columns([2.4, 1.2, 6])
    roi_names = []
    for g in ss.goals:
        if g["ROI"].strip() and g["ROI"] not in roi_names:
            roi_names.append(g["ROI"])
    a1.selectbox("ROI of the new goal", roi_names, index=None, placeholder="pick an ROI or type a new one", accept_new_options=True, key="new_goal_roi", label_visibility="collapsed")
    a2.button("➕ Add goal", on_click=add_goal, key="add_button", help="The goal is inserted after the last goal of its ROI in the file (at the end while the ROI is empty)")


# ----------------------------------------------------------------------------------
# Launcher
# ----------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def edit_clinical_goals(json_path: str | Path) -> bool:
    """Open a web based editor for the clinical goals stored in ``json_path``.

    A local Streamlit server is started and the editor opens in the default web
    browser.  The page lets the user add, copy, delete and modify clinical goals.
    The table is displayed grouped by ROI (alphabetically, case-insensitive) while
    the file order, which the evaluator uses for the cumulative passing rates, is
    preserved: a new goal is inserted after the last goal of its ROI in the file, a
    copy right after its source, and a goal whose ROI is edited keeps its place in
    the file but is displayed with its new group.  The goal type is chosen from a
    drop-down; for *dose at volume* and *volume at dose* goals a tick box selects
    whether the volume is given in cc (ticked) or in % (default).  Dose is in Gy,
    priority is an integer and the *Probabilistic* tick box marks the goals that
    are evaluated over the scenarios.  *Save* writes the goals back to the same
    JSON file in the standard architecture.

    The call blocks until the user presses *Close* in the page (or closes the
    browser tab) and returns ``True`` when the file was saved at least once during
    the session, ``False`` otherwise.

    Parameters
    ----------
    json_path : str or Path
        Path to the clinical goals JSON file.  It must exist and contain a list of
        goals; an empty list (``[]``) is accepted to start from scratch.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Clinical goals file not found: {path}")
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(f"The clinical goals editor needs Streamlit, which is not installed for {sys.executable}")

    with tempfile.TemporaryDirectory(prefix="clinical_goals_editor_") as tmp:
        state_file = Path(tmp) / "state.json"
        env = dict(os.environ)
        env.setdefault("STREAMLIT_SERVER_HEADLESS", "false")  # opens the browser
        env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
            "--server.port", str(_free_port()),
            "--server.address", "127.0.0.1",
            # Streamlit asks for an e-mail address in the terminal the first time it runs on a
            # machine and waits for the answer; without a console (IDE, double-click) it never
            # gets one and the server dies before the editor opens.  Disable the prompt.
            "--server.showEmailPrompt", "false",
            "--browser.gatherUsageStats", "false",
            "--", str(path.resolve()), str(state_file),
        ]
        # no stdin: the server must never wait for keyboard input
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
        started = time.time()
        try:
            while proc.poll() is None:
                time.sleep(0.5)
                try:
                    state = json.loads(state_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
                if state.get("closed"):
                    break
                alive = state.get("alive")
                if alive is not None and time.time() - alive > _STALE_SECONDS:
                    break  # browser tab closed
                if alive is None and time.time() - started > _BROWSER_GRACE_SECONDS:
                    break  # browser never connected
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if not state.get("closed") and state.get("alive") is None:
            # the server stopped (or never answered) before a browser connected: say so
            # instead of silently returning as if the user had closed the editor
            raise RuntimeError(
                f"The clinical goals editor did not start (Streamlit exited with code {proc.returncode}); "
                "see the messages printed above by: " + " ".join(cmd)
            )
        return bool(state.get("saved", False))


def _inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    args = sys.argv[1:]
    if _inside_streamlit():  # started through `streamlit run`
        if not args:
            sys.exit("usage: streamlit run clinicalGoalsEditor.py -- <clinical_goals.json> [<state file>]")
        _run_app(Path(args[0]), Path(args[1]) if len(args) > 1 else None)
    else:  # started with plain python
        if len(args) != 1:
            sys.exit(f"usage: python {Path(__file__).name} <clinical_goals.json>")
        edit_clinical_goals(args[0])
