import numpy as np


class DoseImage:
    """
    A lightweight RTDOSE image container built directly from pydicom datasets.

    Attributes
    ----------
    imageArray : np.ndarray
        The dose data as a 3D numpy array with axis order (x, y, z), in Gy.
    spacing : np.ndarray
        Voxel spacing in mm, in the order (x, y, z).
    origin : np.ndarray
        Coordinates in mm of the first voxel, in the order (x, y, z).
    gridSize : np.ndarray
        Number of voxels along each axis, in the order (x, y, z).
    seriesInstanceUID : str
        The DICOM SeriesInstanceUID of the RTDOSE.
    """

    def __init__(self, imageArray, spacing, origin, gridSize, seriesInstanceUID=None):
        self.imageArray = imageArray
        self.spacing = np.asarray(spacing, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self.gridSize = np.asarray(gridSize, dtype=int)
        self.seriesInstanceUID = seriesInstanceUID

    def resample(self, newSpacing, newGridSize, newOrigin):
        """
        Resample the dose image in place onto a new grid.

        Parameters
        ----------
        newSpacing : array-like
            Target voxel spacing (x, y, z).
        newGridSize : array-like
            Target grid size (x, y, z).
        newOrigin : array-like
            Target origin (x, y, z).
        """
        from scipy.ndimage import zoom

        zoomFactors = np.asarray(newGridSize, dtype=float) / np.asarray(self.imageArray.shape, dtype=float)
        self.imageArray = zoom(self.imageArray, zoomFactors, order=1)
        self.spacing = np.asarray(newSpacing, dtype=float)
        self.gridSize = np.asarray(newGridSize, dtype=int)
        self.origin = np.asarray(newOrigin, dtype=float)
