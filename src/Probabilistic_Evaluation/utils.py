import numpy as np
import time
import threading
import warnings
from functools import wraps

from skimage.draw import polygon2mask
from scipy.ndimage import shift, gaussian_filter, affine_transform

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


def resample_image(imageArray, spacing, origin, newSpacing, newGridSize, newOrigin, antialias=True):
    """
    Resample a 3D image (x, y, z) onto a new grid.

    Every target voxel centre is mapped to its fractional index in the current
    grid and the image is interpolated there (trilinear). Both ``origin`` and
    ``newOrigin`` are the coordinates of the centre of the first voxel (DICOM
    ImagePositionPatient convention), so target centres lie at
    ``newOrigin + i * newSpacing`` and current centres at ``origin + j * spacing``.
    Target voxels that fall outside the current grid take the value of the
    nearest edge voxel. The grids are axis aligned, so the mapping is a scale
    and an offset per axis, applied with ``affine_transform`` without building
    the coordinates of every target voxel. When the target grid equals the
    current grid, ``imageArray`` itself is returned.

    Along axes where the target spacing is coarser than the current one, point
    sampling would alias the gradients. With ``antialias`` the image is first
    low-pass filtered along those axes with a Gaussian of standard deviation
    ``(ratio - 1) / 2`` current voxels, where ``ratio`` is the downsampling
    factor (the same rule scikit-image uses). Axes that are upsampled or
    unchanged are not filtered.

    Parameters
    ----------
    imageArray : np.ndarray
        3D image with axis order (x, y, z).
    spacing : array-like
        Current voxel spacing in mm (x, y, z).
    origin : array-like
        Current coordinates in mm of the centre of the first voxel (x, y, z).
    newSpacing : array-like
        Target voxel spacing in mm (x, y, z).
    newGridSize : array-like
        Target number of voxels (x, y, z).
    newOrigin : array-like
        Target coordinates in mm of the centre of the first voxel (x, y, z).
    antialias : bool, optional
        Low-pass filter before downsampling. Default True.

    Returns
    -------
    np.ndarray
        Resampled image of shape ``newGridSize`` and the same dtype as ``imageArray``
        (``imageArray`` itself, not a copy, when the grid is unchanged).
    """
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    newSpacing = np.asarray(newSpacing, dtype=float)
    newGridSize = np.asarray(newGridSize, dtype=int)
    newOrigin = np.asarray(newOrigin, dtype=float)

    # same grid up to floating point noise (e.g. from resampling_grid): nothing to interpolate
    if (np.array_equal(newGridSize, imageArray.shape) and np.allclose(newSpacing, spacing, rtol=0, atol=1e-9)
            and np.allclose(newOrigin, origin, rtol=0, atol=1e-6)):
        return imageArray

    image = imageArray
    ratio = newSpacing / spacing
    if antialias and np.any(ratio > 1):
        sigma = np.where(ratio > 1, (ratio - 1) / 2, 0.0)
        image = gaussian_filter(image, sigma=sigma, mode="nearest")

    # target index i along an axis maps to the fractional index ratio * i + (newOrigin - origin) / spacing
    return affine_transform(
        image, ratio, offset=(newOrigin - origin) / spacing, output_shape=tuple(newGridSize.tolist()),
        order=1, mode="nearest", output=imageArray.dtype,
    )


def resampling_grid(gridSize, spacing, origin, newSpacing):
    """
    Target grid for resampling an image onto a new voxel spacing while keeping its physical extent.

    ``origin`` is the coordinate of the centre of the first voxel (DICOM ImagePositionPatient
    convention). The physical extent of the current grid runs from the outer edge of its first voxel
    to the outer edge of its last voxel, i.e. ``gridSize * spacing`` per axis. The returned grid starts
    at that same outer edge, so its first voxel centre lies at ``origin - spacing/2 + newSpacing/2``,
    and it holds ``ceil(gridSize * spacing / newSpacing)`` voxels per axis, so that it covers at least
    the same extent. For ``newSpacing == spacing`` the grid is returned unchanged.

    Use it for every resampling so that dose and masks derived from the same source grid share one
    origin (e.g. a RayStation dose grid resampled to 1 mm and ROI masks requested with the RayStation
    grid corner and 1 mm voxels).

    Parameters
    ----------
    gridSize : array-like
        Current number of voxels (x, y, z).
    spacing : array-like
        Current voxel spacing in mm (x, y, z).
    origin : array-like
        Current coordinates in mm of the centre of the first voxel (x, y, z).
    newSpacing : array-like
        Target voxel spacing in mm (x, y, z).

    Returns
    -------
    newGridSize : np.ndarray of int
        Target number of voxels (x, y, z).
    newOrigin : np.ndarray of float
        Target coordinates in mm of the centre of the first voxel (x, y, z).
    """
    gridSize = np.asarray(gridSize, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    newSpacing = np.asarray(newSpacing, dtype=float)
    if np.any(spacing <= 0) or np.any(newSpacing <= 0):
        raise ValueError("spacing and newSpacing must be strictly positive.")

    extent = gridSize * spacing
    # round before ceil so that exact ratios affected by floating point noise do not gain a voxel
    newGridSize = np.ceil(np.round(extent / newSpacing, 6)).astype(int)
    newOrigin = origin - spacing / 2.0 + newSpacing / 2.0
    return newGridSize, newOrigin


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
    # (ox, oy) is the centre of the first voxel (RayStation Corner + spacing/2) and polygon2mask samples sub-pixel
    # centres at integer coordinates, so the precision x precision sub-pixels of a voxel are centred on its centre
    center_offset = 0.5 * (precision - 1)

    for poly in polygons_xy:
        rows = ((poly[:, 0] - ox) / sx) * precision + center_offset
        cols = ((poly[:, 1] - oy) / sy) * precision + center_offset
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

def _group_polygons_by_z(polygonMesh, z_tolerance: float = 1e-2) -> list:
    """
    Group the polygons of a contour into planes.

    Polygons whose z lies within ``z_tolerance`` mm of the previous one (in z order) belong to the same plane, located
    at the mean z of its polygons.

    Returns
    -------
    list of (float, list of np.ndarray)
        Planes sorted by z, each with its polygons as (N, 2) arrays of x, y coordinates in mm.
    """
    polygons = []
    for contour_data in polygonMesh:
        coords = np.asarray(contour_data, dtype=float)
        if coords.size < 9 or coords.size % 3 != 0:
            continue
        triplets = coords.reshape(-1, 3)
        polygons.append((float(np.median(triplets[:, 2])), triplets[:, :2]))
    polygons.sort(key=lambda polygon: polygon[0])

    planes = []
    for z, xy in polygons:
        if planes and z - planes[-1][0][-1] <= z_tolerance:
            planes[-1][0].append(z)
            planes[-1][1].append(xy)
        else:
            planes.append(([z], [xy]))
    return [(float(np.mean(zs)), polys) for zs, polys in planes]

def _plane_weights(voxel_z, voxel_size, plane_z, cap_below, cap_above) -> np.ndarray:
    """
    Weight of every contour plane in the average of the ROI cross-section over the z extent of each voxel.

    Between two consecutive planes the cross-section is interpolated linearly. Below the first and above the last plane
    it is kept unchanged over ``cap_below`` and ``cap_above`` mm, so that planes a distance d apart hold the volume
    sum(plane area) * d, whatever the voxel size.

    Parameters
    ----------
    voxel_z : np.ndarray
        Coordinates in mm of the voxel centres along z.
    voxel_size : float
        Voxel size in mm along z.
    plane_z : np.ndarray
        Sorted coordinates in mm of the contour planes.
    cap_below, cap_above : float
        Extent in mm of the cross-section below the first and above the last plane.

    Returns
    -------
    np.ndarray
        Weights of shape (len(voxel_z), len(plane_z)).
    """
    low, high = voxel_z - voxel_size / 2.0, voxel_z + voxel_size / 2.0
    weights = np.zeros((len(voxel_z), len(plane_z)))
    weights[:, 0] += np.clip(np.minimum(high, plane_z[0]) - np.maximum(low, plane_z[0] - cap_below), 0.0, None)
    weights[:, -1] += np.clip(np.minimum(high, plane_z[-1] + cap_above) - np.maximum(low, plane_z[-1]), 0.0, None)
    for i in range(len(plane_z) - 1):
        z0, z1 = plane_z[i], plane_z[i + 1]
        a, b = np.clip(low, z0, z1), np.clip(high, z0, z1)
        # integral over [a, b] of the linear weight (z - z0) / (z1 - z0) of the upper plane
        upper = ((b - z0) ** 2 - (a - z0) ** 2) / (2.0 * (z1 - z0))
        weights[:, i] += (b - a) - upper
        weights[:, i + 1] += upper
    return weights / voxel_size

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

    Every contour plane is rasterized in-plane with supersampling. Along z, the cross-section is interpolated linearly
    between consecutive planes and kept unchanged over half the distance to the neighbouring plane beyond the first and
    last planes, and every voxel receives the average over its z extent. The mask therefore holds the volume
    sum(plane area * plane distance) at the position of the contours, whatever the voxel size.

    Parameters
    ---------
    origin: array
        Coordinates in mm of the centre of the first voxel of the output mask image (DICOM-like origin,
        i.e. RayStation Corner + spacing/2), as returned by resampling_grid for the dose grid.
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

    planes = _group_polygons_by_z(contour.polygonMesh)
    if not planes:
        return np.zeros(tuple(gridSize.tolist()), dtype=np.float32)
    plane_z = np.array([z for z, _ in planes])
    xy = np.concatenate([polygon for _, polygons in planes for polygon in polygons])

    # every plane covers half the distance to its neighbours; a lone plane covers one voxel
    cap_below = 0.5 * (plane_z[1] - plane_z[0]) if len(planes) > 1 else 0.5 * float(spacing[2])
    cap_above = 0.5 * (plane_z[-1] - plane_z[-2]) if len(planes) > 1 else 0.5 * float(spacing[2])

    contour_min = np.array([xy[:, 0].min(), xy[:, 1].min(), plane_z[0] - cap_below], dtype=float)
    contour_max = np.array([xy[:, 0].max(), xy[:, 1].max(), plane_z[-1] + cap_above], dtype=float)

    box_start_idx = np.floor((contour_min - origin) / spacing - 0.5).astype(int)
    box_end_idx = np.ceil((contour_max - origin) / spacing + 0.5).astype(int)

    box_start_idx = np.maximum(box_start_idx, 0)
    box_end_idx = np.minimum(box_end_idx, gridSize - 1)
    if np.any(box_end_idx < box_start_idx):
        return np.zeros(tuple(gridSize.tolist()), dtype=np.float32)

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

    # every voxel holds the z-average of the cross-section over its extent: a weighted sum of the nearby rasterized planes
    weights = _plane_weights(box_origin[2] + np.arange(gz) * spacing[2], float(spacing[2]), plane_z, cap_below, cap_above)
    plane_masks = {}
    for k in range(gz):
        for i in np.flatnonzero(weights[k] > 0):
            if i not in plane_masks:
                plane_masks[i] = _polygon_to_mask_slice(
                    polygons_xy=planes[i][1],
                    contour_origin_xy=box_origin[:2],
                    contour_spacing_xy=spacing[:2],
                    grid_xy=(gx, gy),
                    precision=adaptive_precision,
                    combine_mode="xor",
                )
            local_mask3D[:, :, k] += np.float32(weights[k, i]) * plane_masks[i]

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
