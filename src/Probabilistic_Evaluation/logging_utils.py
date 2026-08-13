# Probabilistic_Evaluation/logging_utils.py
import logging
import os
import sys
import platform
import importlib.metadata
import functools
import inspect
import threading
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

_LOGGER_NAME = "ProbEval"
logger = logging.getLogger(_LOGGER_NAME)


def enable_logging(to_file: bool = False, log_dir: str = ".\\Logs",
                    console_output: bool = True,
                    console_level: int = logging.INFO,
                    file_level: int = logging.DEBUG):
    """
    Call once at the start of a user script.

    console_output: if True, INFO+ (by default) prints to terminal.
    to_file: if True, DEBUG+ (everything) is written to a timestamped .log file.
    """
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False  # stop double-printing via root logger (Jupyter etc.)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)-15s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(console_level)
        console.setFormatter(fmt)
        logger.addHandler(console)

    if to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"Log_file.log"
        Path(log_path).unlink(missing_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        _log_environment_header()
        _install_exception_hook()
        logger.info(f"Logging to file: {log_path.resolve()}")

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def _log_environment_header():
    logger.info("=" * 60)
    logger.info("ProbEval session started")
    logger.info(f"ProbEval version   : {_safe_version('Probabilistic_Evaluation')}")
    logger.info(f"Python version  : {platform.python_version()}")
    logger.info(f"Platform        : {platform.platform()}")
    for dep in ("numpy", "scipy", "pydicom"):
        logger.info(f"{dep:<15} : {_safe_version(dep)}")
    logger.info("=" * 60)


def _safe_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _install_exception_hook():
    """Uncaught exceptions in the MAIN thread land in the log file with traceback."""
    def hook(exc_type, exc_value, exc_tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = hook


def log_call(level=logging.DEBUG, log_result=False, log_duration=True):
    """Decorator: logs function entry (args), exit, duration, and exceptions."""
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fn_logger = logger.getChild(func.__module__)

            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                params_str = ", ".join(f"{k}={_safe_repr(v)}" for k, v in bound.arguments.items())
            except TypeError:
                params_str = f"args={args}, kwargs={kwargs}"

            fn_logger.log(level, f"CALL  {func.__qualname__}({params_str})")

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                fn_logger.exception(f"FAIL  {func.__qualname__}")
                raise
            else:
                duration = time.perf_counter() - start
                msg = f"DONE  {func.__qualname__}"
                if log_duration:
                    msg += f" [{duration:.3f}s]"
                if log_result:
                    msg += f" -> {_safe_repr(result)}"
                fn_logger.log(level, msg)
                return result

        return wrapper
    return decorator


def _safe_repr(value, max_len=200):
    if hasattr(value, "shape"):
        return f"<{type(value).__name__} shape={value.shape} dtype={getattr(value, 'dtype', '?')}>"
    r = repr(value)
    return r if len(r) <= max_len else r[:max_len] + "...>"


class GroupedMemoryHandler(logging.Handler):
    """
    Buffers log records in memory, grouped by thread name.
    Attach during a multithreaded section instead of the normal file handler,
    then call flush_grouped() in a finally block to write everything out
    grouped by thread, even if the run crashed.
    """
    def __init__(self, target_handler):
        super().__init__()
        self.target_handler = target_handler
        self._buffers = defaultdict(list)
        self._lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            self._buffers[record.threadName].append(record)

    def flush_grouped(self, main_thread_name="MainThread"):
        with self._lock:
            ordered_keys = [k for k in self._buffers if k == main_thread_name]
            ordered_keys += sorted(k for k in self._buffers if k != main_thread_name)

            for thread_name in ordered_keys:
                header = logging.LogRecord(
                    name=self.target_handler.name or "grouped",
                    level=logging.INFO, pathname="", lineno=0,
                    msg=f"\n--- Thread: {thread_name} ---", args=None, exc_info=None,
                )
                self.target_handler.emit(header)
                for record in self._buffers[thread_name]:
                    self.target_handler.emit(record)

            self._buffers.clear()