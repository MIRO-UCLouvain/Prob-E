class ROIContour:
    """
    The contour data of a single ROI extracted from an RTSTRUCT DICOM file.

    Attributes
    ----------
    name : str
        The ROI name.
    polygonMesh : list of np.ndarray
        One flat array per contour ring, in the DICOM ContourData format
        [x0, y0, z0, x1, y1, z1, ...].
    """

    def __init__(self, name, polygonMesh):
        self.name = name
        self.polygonMesh = polygonMesh
