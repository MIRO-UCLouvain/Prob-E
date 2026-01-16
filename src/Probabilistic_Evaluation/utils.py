import numpy as np

def shift_dose_image(doseImage, shift):
    """
    Shift the dose image by the specified amount in each dimension.
    Parameters
    ----------
    doseImage : np.ndarray
        The dose image to be shifted.
    shift : tuple
        A tuple specifying the shift in each dimension (z, y, x).
    Returns
    -------
    shifted_dose : np.ndarray
        The shifted dose image.
    """
    if len(shift) != doseImage.ndim:
        raise ValueError("Shift dimensions must match dose image dimensions.")
    shifted_dose = np.roll(doseImage, shift=shift, axis=(0, 1, 2))
    return shifted_dose

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
