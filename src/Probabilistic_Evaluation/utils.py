import numpy as np
import time
import threading
import warnings
from functools import wraps

from skimage.draw import polygon2mask
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


def _round_half_away_from_zero(value: float) -> int:
    """Round half values away from zero to avoid banker's rounding shifts."""
    if value >= 0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))

def _polygon_to_mask_slice(
    polygons_xy,
    contour_origin_xy,
    contour_spacing_xy,
    grid_xy,
    precision: int,
    combine_mode: str = "xor",
) -> np.ndarray:
    """Convert polygons on one z slice into a partial-volume mask."""
    gx, gy = int(grid_xy[0]), int(grid_xy[1])
    hr_shape = (gx * precision, gy * precision)
    slice_mask = np.zeros((gx, gy), dtype=np.float32)

    ox, oy = float(contour_origin_xy[0]), float(contour_origin_xy[1])
    sx, sy = float(contour_spacing_xy[0]), float(contour_spacing_xy[1])
    shift_x = 0.5 * (1.0 - sx)
    shift_y = 0.5 * (1.0 - sy)
    center_offset = 0.5 * (precision - 1)

    for poly in polygons_xy:
        rows = ((poly[:, 0] + shift_x - ox) / sx) * precision + center_offset
        cols = ((poly[:, 1] + shift_y - oy) / sy) * precision + center_offset
        polygon_mask = polygon2mask(hr_shape, np.column_stack((rows, cols)))

        if precision == 1:
            down = polygon_mask.astype(np.float32)
        else:
            down = polygon_mask.astype(np.float32).reshape(gx, precision, gy, precision).mean(axis=(1, 3))

        if combine_mode == "union":
            slice_mask = np.maximum(slice_mask, down)
        else:
            slice_mask = np.abs(slice_mask - down)

    return np.clip(slice_mask, 0.0, 1.0)

def _group_polygons_by_z(polygonMesh, z_tolerance: float = 1e-3) -> dict:
    """Group polygons by z coordinate with tolerance clustering."""
    grouped = {}
    for contour_data in polygonMesh:
        coords = np.asarray(contour_data, dtype=float)
        if coords.size < 9 or coords.size % 3 != 0:
            continue
        triplets = coords.reshape(-1, 3)
        z = float(np.median(triplets[:, 2]))
        if z_tolerance > 0:
            z = float(np.round(z / z_tolerance) * z_tolerance)
        grouped.setdefault(z, []).append(triplets[:, :2])
    return grouped

def _interpolate_between_two_slices(lower_slice: np.ndarray, upper_slice: np.ndarray) -> np.ndarray:
    """Blend two slices, using non-zero values when only one side is present."""
    lower = np.clip(lower_slice, 0.0, 1.0).astype(np.float32)
    upper = np.clip(upper_slice, 0.0, 1.0).astype(np.float32)

    out = 0.5 * (lower + upper)
    eps = 1e-8

    lower_zero = np.abs(lower) <= eps
    upper_zero = np.abs(upper) <= eps
    out[lower_zero & (~upper_zero)] = upper[lower_zero & (~upper_zero)]
    out[upper_zero & (~lower_zero)] = lower[upper_zero & (~lower_zero)]

    lower_one = np.abs(lower - 1.0) <= eps
    upper_one = np.abs(upper - 1.0) <= eps
    out[lower_one | upper_one] = 1.0

    return np.clip(out, 0.0, 1.0).astype(np.float32)

def get_partial_volume_mask(
    contour,
    origin,
    gridSize,
    spacing,
    precision=16,
    binarization_threshold=None,
):
    """
    Convert the ROI contour to a partial-volume mask image.

    Parameters
    ---------
    origin: array
        Origin of the output mask image.
    gridSize: array
        Grid size of the output mask image.
    spacing: array
        Voxel spacing of the output mask image.
    precision: int (optional)
        Supersampling factor used during rasterization. Higher values improve
        partial-volume accuracy but increase computation time.
    binarization_threshold: float (optional)
        If provided, convert the partial-volume mask to a binary mask using
        this threshold in [0.0, 1.0].

    Returns
    -------
    mask: ROIMask
        Partial-volume ROI mask (or binary mask when
        ``binarization_threshold`` is provided).
    """

    if contour is None or not hasattr(contour, "polygonMesh"):
        raise ValueError("contour must provide a polygonMesh attribute.")

    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    if precision is None or int(precision) <= 0:
        raise ValueError("precision must be a positive integer.")
    precision = int(precision)

    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    gridSize = np.asarray(gridSize, dtype=int)

    if spacing.shape[0] != 3 or origin.shape[0] != 3 or gridSize.shape[0] != 3:
        raise ValueError("origin, spacing and gridSize must be 3D.")
    if np.any(spacing <= 0):
        raise ValueError("spacing values must be strictly positive.")
    if np.any(gridSize <= 0):
        raise ValueError("gridSize values must be strictly positive.")

    allX = []
    allY = []
    allZ = []
    for contourData in contour.polygonMesh:
        coords = np.asarray(contourData, dtype=float)
        if coords.size < 9 or coords.size % 3 != 0:
            continue
        allX.append(coords[0::3])
        allY.append(coords[1::3])
        allZ.append(float(np.mean(coords[2::3])))

    allX = np.sort(np.concatenate(allX).astype(float))
    allY = np.sort(np.concatenate(allY).astype(float))
    allZ = np.sort(np.asarray(allZ, dtype=float))

    contour_min = np.array([allX[0], allY[0], allZ[0]], dtype=float)
    contour_max = np.array([allX[-1], allY[-1], allZ[-1]], dtype=float)

    zDiff = np.abs(np.diff(allZ))
    zDiff[zDiff == 0] = np.inf
    finite_zDiff = zDiff[np.isfinite(zDiff)]
    native_z_spacing = float(finite_zDiff.min()) if finite_zDiff.size > 0 else float(spacing[2])

    box_start_idx = np.floor((contour_min - origin) / spacing - 0.5).astype(int)
    box_end_idx = np.ceil((contour_max - origin) / spacing + 0.5).astype(int)

    if float(spacing[2]) < native_z_spacing:
        box_end_idx[2] += 1

    box_start_idx = np.maximum(box_start_idx, 0)
    box_end_idx = np.minimum(box_end_idx, gridSize - 1)

    box_grid_size = (box_end_idx - box_start_idx + 1).astype(int)
    box_origin = origin + box_start_idx.astype(float) * spacing

    gx, gy, gz = int(box_grid_size[0]), int(box_grid_size[1]), int(box_grid_size[2])
    local_mask3D = np.zeros((gx, gy, gz), dtype=np.float32)

    max_high_res_side = 512
    if max(gx, gy) >= 256:
        adaptive_precision = 1
    else:
        adaptive_precision = max(
            1,
            min(
                int(precision),
                int(max_high_res_side / max(gx, gy)) if max(gx, gy) > 0 else int(precision),
            ),
        )

    z_group_tol = max(1e-4, 0.02 * float(spacing[2]))
    grouped = _group_polygons_by_z(contour.polygonMesh, z_tolerance=z_group_tol)

    interpolate_z = float(spacing[2]) < native_z_spacing

    if interpolate_z:
        grouped_by_k = {}
        for z_val, polys in grouped.items():
            k_global = _round_half_away_from_zero((float(z_val) - float(origin[2])) / float(spacing[2]))
            k_local = int(k_global - int(box_start_idx[2]))
            if 0 <= k_local < gz:
                grouped_by_k.setdefault(k_local, []).extend(polys)

        contour_k = np.array(sorted(grouped_by_k.keys()), dtype=int)
        first_k = int(contour_k[0])
        last_k = int(contour_k[-1])
        native_steps = max(1.0, float(native_z_spacing) / float(spacing[2]))
        hat_half_steps = 0.5 * native_steps
        exact_slice_cache = {}

        for k_exact in contour_k:
            local_mask3D[:, :, int(k_exact)] = _polygon_to_mask_slice(
                polygons_xy=grouped_by_k[int(k_exact)],
                contour_origin_xy=box_origin[:2],
                contour_spacing_xy=spacing[:2],
                grid_xy=(gx, gy),
                precision=adaptive_precision,
                combine_mode="xor",
            )
            exact_slice_cache[int(k_exact)] = local_mask3D[:, :, int(k_exact)].copy()

        for k in range(gz):
            if k in grouped_by_k:
                continue

            lower_candidates = contour_k[contour_k < k]
            upper_candidates = contour_k[contour_k > k]

            if lower_candidates.size == 0:
                d = float(first_k - k)
                if d <= hat_half_steps:
                    t_hat = d / max(hat_half_steps, 1e-6)
                    local_mask3D[:, :, k] = (1.0 - t_hat) * exact_slice_cache[first_k]
                continue

            if upper_candidates.size == 0:
                d = float(k - last_k)
                if d <= hat_half_steps:
                    t_hat = d / max(hat_half_steps, 1e-6)
                    local_mask3D[:, :, k] = (1.0 - t_hat) * exact_slice_cache[last_k]
                continue

            k0 = int(lower_candidates[-1])
            k1 = int(upper_candidates[0])
            if k1 <= k0:
                continue

            local_mask3D[:, :, k] = _interpolate_between_two_slices(
                exact_slice_cache[k0],
                exact_slice_cache[k1],
            )

        for k_exact, exact_slice in exact_slice_cache.items():
            local_mask3D[:, :, k_exact] = exact_slice
    else:
        grouped_z = np.array(sorted(grouped.keys()), dtype=float)
        contour_k_coarse = []
        for z_val in grouped_z:
            k_global = _round_half_away_from_zero((float(z_val) - float(origin[2])) / float(spacing[2]))
            k_local = int(k_global - int(box_start_idx[2]))
            if 0 <= k_local < gz:
                contour_k_coarse.append(k_local)

        coarse_precision = max(1, min(adaptive_precision, 4))
        coarse_mask_cache = {}
        first_k_coarse = int(min(contour_k_coarse))
        last_k_coarse = int(max(contour_k_coarse))

        for k in range(gz):
            if k < first_k_coarse or k > last_k_coarse:
                continue
            z_target = float(box_origin[2]) + k * float(spacing[2])
            z_sel = float(grouped_z[int(np.argmin(np.abs(grouped_z - z_target)))])
            polys = grouped[z_sel]
            if not polys:
                continue
            cache_key = float(z_sel)
            cached_mask = coarse_mask_cache.get(cache_key)
            if cached_mask is None:
                cached_mask = _polygon_to_mask_slice(
                    polygons_xy=polys,
                    contour_origin_xy=box_origin[:2],
                    contour_spacing_xy=spacing[:2],
                    grid_xy=(gx, gy),
                    precision=coarse_precision,
                    combine_mode="xor",
                )
                coarse_mask_cache[cache_key] = cached_mask
            local_mask3D[:, :, k] = cached_mask

    if binarization_threshold is not None:
        if not (0.0 <= float(binarization_threshold) <= 1.0):
            raise ValueError("binarization_threshold must be in the range [0.0, 1.0].")
        local_mask3D = (local_mask3D >= float(binarization_threshold)).astype(bool)


    # Insert local aligned box directly into requested grid.
    full_mask = np.zeros(tuple(gridSize.tolist()), dtype=np.float32)
    x0, y0, z0 = int(box_start_idx[0]), int(box_start_idx[1]), int(box_start_idx[2])
    x1, y1, z1 = x0 + gx, y0 + gy, z0 + gz
    full_mask[x0:x1, y0:y1, z0:z1] = local_mask3D

    return np.clip(full_mask.astype(np.float32), 0.0, 1.0)
