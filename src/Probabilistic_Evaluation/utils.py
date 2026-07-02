import numpy as np
import time
import threading
from functools import wraps
from scipy.ndimage import shift

from Probabilistic_Evaluation.logging_utils import log_call, logger

def shift_dose_image(doseImage, shift):
    """
    Shift the dose image by the specified amount in each dimension.

    Parameters
    ----------
    doseImage : np.ndarray
        The dose image to be shifted.
    shift : tuple
        A tuple specifying the shift in each dimension (x,y,z).

    Returns
    ------- 
    shifted_dose : np.ndarray
        The shifted dose image.
    """
    logger.debug(f"Shifting dose image by {shift} voxels.")
    
    shift = (int(round(shift[0])), int(round(shift[1])), int(round(shift[2])))
    if len(shift) != doseImage.ndim:
        raise ValueError("Shift dimensions must match dose image dimensions.")
    shifted_dose = np.roll(doseImage, shift=shift, axis=(0, 1, 2))
    return shifted_dose

def shift_dose_for_enhanced_sampling(doseImage):
        shift_vec = [0.5, 0.5, 0.5]  # Shift by half a voxel in each dimension
        logger.debug(f"Shifting dose image for enhanced sampling by {shift_vec} voxels.")
        half_shifted_dose = shift(doseImage, shift=shift_vec, order=1, mode='constant' )  
        return half_shifted_dose


def linearInterpolator(x: float, x_array: np.ndarray, y_array: np.ndarray) -> float:
    """
    Linear interpolation helper.

    Parameters
    ----------
    x : float
        Target x value.
    x_array : np.ndarray
        Monotonic array of x values.
    y_array : np.ndarray
        Array of y values corresponding to x_array.

    Returns
    -------
    float
        Interpolated y value at x.
    """
    if x <= x_array[0]:
        return y_array[0]
    if x >= x_array[-1]:
        return y_array[-1]

    idx = np.searchsorted(x_array, x) - 1

    x0, x1 = x_array[idx], x_array[idx + 1]
    y0, y1 = y_array[idx], y_array[idx + 1]

    if x1 == x0:
        return y0

    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)



class Timer:
    """
    A singleton timer class for measuring function execution time with multi-threading support.

    This class records timing information for functions, supporting nested calls and
    multi-threaded execution. It maintains separate timing data for the main thread and
    worker threads, and provides a formatted report of timing statistics.

    Attributes
    ----------
    times : dict
        Dictionary storing timing data keyed by function label.
    thread_totals : dict
        Dictionary storing total execution time per thread.
    """
    _instance = None
    _local = threading.local()
    _lock = threading.Lock()
    times = {}
    thread_totals = {}
    _main_thread_id = threading.main_thread().ident

    @classmethod
    def get(cls):
        """
        Get the singleton Timer instance.

        Returns
        -------
        Timer
            The singleton Timer instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def call_stack(self):
        """
        Get the call stack for the current thread.

        Returns
        -------
        list
            The call stack containing labels of nested function calls for the current thread.
        """
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def _is_main_thread(self):
        """
        Check if the current thread is the main thread.

        Returns
        -------
        bool
            True if the current thread is the main thread, False otherwise.
        """
        return threading.current_thread().ident == self._main_thread_id

    def record(self, label, elapsed, depth):
        """
        Record the execution time of a function.

        Parameters
        ----------
        label : str
            The label identifying the function (usually "ClassName.method_name" or "function_name").
        elapsed : float
            The elapsed time in seconds.
        depth : int
            The depth of the call in the call stack (1 for top-level calls).

        Returns
        -------
        None
        """
        thread = threading.current_thread()
        is_main = self._is_main_thread()

        with self._lock:
            key = label if is_main else f"__thread__{thread.name}__{label}"
            self.times.setdefault(key, {"total": 0, "calls": 0, "is_main": is_main, "thread": thread.name})
            self.times[key]["total"] += elapsed
            self.times[key]["calls"] += 1

            if depth == 1:
                self.thread_totals.setdefault(thread.name, 0)
                self.thread_totals[thread.name] += elapsed

    def report(self):
        """
        Print a formatted timing report of all recorded function executions.

        The report displays separate sections for main thread and worker thread statistics,
        including total execution time, number of calls, and average times for worker threads.

        Returns
        -------
        None
        """
        with self._lock:
            main_entries = {k: v for k, v in self.times.items() if v["is_main"]}
            worker_entries = {}
            for k, v in self.times.items():
                if not v["is_main"]:
                    label = k.split(f"__thread__{v['thread']}__", 1)[1]
                    worker_entries.setdefault(label, []).append(v)

        all_labels = list(main_entries.keys()) + list(worker_entries.keys()) + ["TOTAL"]
        func_width = max(50, max((len(label) for label in all_labels), default=0))

        main_header = f"  {'Function':<{func_width}} {'total':>8}  {'calls':>5}"
        main_sep = f"  {'-' * func_width} {'-' * 8}  {'-' * 5}"
        worker_header = f"  {'Function':<{func_width}} {'total':>8}  {'calls':>5} {'thr':>4}"
        worker_sep = f"  {'-' * func_width} {'-' * 8}  {'-' * 5} {'-' * 4}"

        inner_width = max(len(main_header), len(worker_header), len(" Main Thread"), len(" Worker Threads (Averaged)"))

        def framed(line):
            return f"║{line:<{inner_width}}║"

        print("\n" + "╔" + "═" * inner_width + "╗")
        print(framed(f"{'Timing Report':^{inner_width}}"))
        print("╠" + "═" * inner_width + "╣")

        # main thread
        if main_entries:
            print(framed(" Main Thread"))
            print(framed(main_header))
            print(framed(main_sep))
            main_thread_name = threading.main_thread().name
            real_total = self.thread_totals.get(main_thread_name, sum(t["total"] for t in main_entries.values()))
            for key, t in sorted(main_entries.items()):
                row = f"  {key:<{func_width}} {t['total']:>7.4f}s {t['calls']:>5}"
                print(framed(row))
            total_row = f"  {'TOTAL':<{func_width}} {real_total:>8.4f}s"
            print(framed(total_row))

        # worker threads averaged across all worker threads
        if worker_entries:
            print("╠" + "═" * inner_width + "╣")
            print(framed(" Worker Threads (Averaged)"))
            print(framed(worker_header))
            print(framed(worker_sep))

            averaged_entries = {}
            for label, entries in worker_entries.items():
                averaged_entries[label] = {
                    "total": sum(t["total"] for t in entries) / len(entries),
                    "calls": sum(t["calls"] for t in entries) / len(entries),
                    "threads": len(entries),
                }

            worker_thread_names = sorted(set(t["thread"] for t in self.times.values() if not t["is_main"]))
            real_total = (
                sum(self.thread_totals.get(thread_name, 0) for thread_name in worker_thread_names) / len(worker_thread_names)
                if worker_thread_names
                else 0
            )
            for label, t in sorted(averaged_entries.items()):
                row = f"  {label:<{func_width}} {t['total']:>7.4f}s {t['calls']:>5.1f} {t['threads']:>4}"
                print(framed(row))
            total_row = f"  {'TOTAL':<{func_width}} {real_total:>8.4f}s"
            print(framed(total_row))

        print("╚" + "═" * inner_width + "╝")

    def reset(self):
        """
        Reset all timing data.

        Clears the times and thread_totals dictionaries to start fresh timing measurements.

        Returns
        -------
        None
        """
        with self._lock:
            self.times = {}
            self.thread_totals = {}


def timed(func):
    """
    Decorator to measure and record the execution time of a function.

    This decorator automatically times function calls and records the results in the global
    Timer instance. It supports nested calls and multi-threaded execution. The timing data
    can be accessed via Timer.get().report().

    Parameters
    ----------
    func : callable
        The function to be timed.

    Returns
    -------
    callable
        The wrapped function that records execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        label = f"{args[0].__class__.__name__}.{func.__name__}" if args else func.__name__
        timer = Timer.get()
        timer.call_stack.append(label)
        depth = len(timer.call_stack)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            timer.call_stack.pop()
            timer.record(label, elapsed, depth)
        return result
    return wrapper