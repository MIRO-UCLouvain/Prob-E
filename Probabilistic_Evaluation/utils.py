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
    if doseImage.shape != shift:
        raise ValueError("Shift dimensions must match dose image dimensions.")
    shifted_dose = np.roll(doseImage, shift=shift, axis=(0, 1, 2))
    return shifted_dose