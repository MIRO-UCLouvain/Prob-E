import numpy as np

def shift_dose_image(doseImage, shift):
        shifted_dose = np.roll(doseImage, shift=shift, axis=(0, 1, 2))
        return shifted_dose