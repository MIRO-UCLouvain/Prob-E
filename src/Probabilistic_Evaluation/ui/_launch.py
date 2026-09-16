"""Shared subprocess launcher for the optional Streamlit-based UI tools.

Both :mod:`Probabilistic_Evaluation.ui.clinicalGoalsEditor` and
:mod:`Probabilistic_Evaluation.ui.resultsViewer` run their page as a
``streamlit run`` subprocess and need to know when the user closed the
browser tab, so they can return control to the caller. This module holds
the machinery both share: a small on-disk state file the page's liveness
thread writes a heartbeat to while a browser is connected, and the
launcher-side polling loop that waits for that heartbeat to go stale (or
for the page to report itself closed) before returning.

This module only depends on the standard library and Streamlit (imported
lazily, only where actually needed), so importing it never requires the
``ui`` extra to be installed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Heartbeat written by the page while a browser is connected; the launcher treats the
# page as closed once it goes stale.
_HEARTBEAT_SECONDS = 1
_STALE_SECONDS = 15
_BROWSER_GRACE_SECONDS = 180
_LIVENESS_THREAD = "streamlit-app-liveness"

# The package logger (see logging_utils), looked up by name so that the package itself is not imported.
logger = logging.getLogger("ProbEval")


def require_streamlit(feature: str) -> None:
    """Raise a clear ``RuntimeError`` unless Streamlit (the ``ui`` extra) is installed."""
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError(
            f"{feature} needs the 'ui' extra, which is not installed for {sys.executable}. "
            "Install it with: pip install rt-probabilistic-evaluation[ui]"
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_state(state_file: Path | None, **updates) -> None:
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


def start_liveness_thread(state_file: Path | None) -> None:
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
                write_state(state_file, alive=time.time())
            time.sleep(_HEARTBEAT_SECONDS)

    threading.Thread(target=loop, name=_LIVENESS_THREAD, daemon=True).start()


def run_streamlit_script(script_path: Path, args: list[str]) -> dict:
    """Run one Streamlit page as a subprocess until its browser tab is closed.

    Spawns ``streamlit run script_path -- <args> <state file>`` (a temporary
    state-file path is appended as the last positional argument, for the page
    to poll/write via :func:`write_state` / :func:`start_liveness_thread`).

    Blocks until the page's liveness heartbeat goes stale, the page reports
    itself closed, or no browser ever connects. Raises ``RuntimeError`` in
    that last case (the page failed to start or nobody ever opened it).
    Returns the final state dict written by the page, whatever fields it
    happens to contain (e.g. ``{"saved": True, "closed": True}``).
    """
    with tempfile.TemporaryDirectory(prefix="streamlit_app_") as tmp:
        state_file = Path(tmp) / "state.json"
        env = dict(os.environ)
        env.setdefault("STREAMLIT_SERVER_HEADLESS", "false")  # opens the browser
        env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(script_path.resolve()),
            "--server.port", str(_free_port()),
            "--server.address", "127.0.0.1",
            # Streamlit asks for an e-mail address in the terminal the first time it runs on a
            # machine and waits for the answer; without a console (IDE, double-click) it never
            # gets one and the server dies before the page opens. Disable the prompt.
            "--server.showEmailPrompt", "false",
            "--browser.gatherUsageStats", "false",
            "--", *args, str(state_file),
        ]
        # no stdin: the server must never wait for keyboard input
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
        logger.debug(f"Started the Streamlit server for {script_path.name} (pid {proc.pid}): {' '.join(cmd)}")
        started = time.time()
        try:
            while proc.poll() is None:
                time.sleep(0.5)
                try:
                    state = json.loads(state_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
                if state.get("closed"):
                    logger.debug(f"{script_path.name}: the page reported that it was closed.")
                    break
                alive = state.get("alive")
                if alive is not None and time.time() - alive > _STALE_SECONDS:
                    logger.debug(f"{script_path.name}: no heartbeat for {_STALE_SECONDS}s, the browser tab was closed.")
                    break  # browser tab closed
                if alive is None and time.time() - started > _BROWSER_GRACE_SECONDS:
                    logger.debug(f"{script_path.name}: no browser connected within {_BROWSER_GRACE_SECONDS}s.")
                    break  # browser never connected
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.debug(f"{script_path.name}: the Streamlit server (pid {proc.pid}) did not stop within 10s and is killed.")
                    proc.kill()
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        logger.debug(f"{script_path.name}: the Streamlit server stopped (exit code {proc.returncode}), final page state {state}.")
        if not state.get("closed") and state.get("alive") is None:
            # the server stopped (or never answered) before a browser connected: say so
            # instead of silently returning as if the user had closed the page
            raise RuntimeError(
                f"The Streamlit page did not start (streamlit exited with code {proc.returncode}); "
                "see the messages printed above by: " + " ".join(cmd)
            )
        return state
