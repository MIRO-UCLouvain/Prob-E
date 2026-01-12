import numpy as np
import matplotlib.pyplot as plt

class DVH():
    """
    Class to compute Dose-Volume Histogram (DVH) from dose distribution and a mask.
    Attributes:
        dosemap (np.ndarray): 3D array representing the dose distribution.  
        mask (np.ndarray): Boolean mask for the region of interest.
        dvh (np.ndarray): Computed DVH for the mask.
        maxDVH (float): Maximum dose value for DVH calculation.
    Methods:
        computeDVH(): Computes the DVH for the given mask.
        computeDx(x): Computes the dose at which x% of the volume receives at least that dose.
        
    """


    def __innit__(self, dosemap:np.ndarray, mask:np.ndarray, max_DVH:float=100.0) :
        self.dosemap = dosemap
        self.mask = mask
        self.bin_dose = None # Store bin doses for Dx and Vx calculations, makes Dx calculations easier
        self.dvh = None
        self.DMin = None
        self.DMax = None
        self.DMean = None
      
        self.maxDVH = max_DVH
        
        self.computeDVH()
    

    def computeDVH(self):
        """
        Compute Dose-Volume Histogram (DVH) for a given mask.

        """
        n_bins = 4096
        mask = self.mask.astype(bool)
        dose_mask = self.dosemap[mask]
        bin_size = self.maxDVH / n_bins
        bin_edges = np.arange(0, self.maxDVH + 0.5*bin_size, bin_size) # np.arange is exclusive right limit
        bin_edges[-1] += dose_mask.max()  # Ensure the max dose is included in the last bin
        hist, _ = np.histogram(dose_mask, bins=bin_edges)
        hist = np.flip(hist,0)  # Flip to get descending order
        dvh = np.cumsum(hist) # Cumulative sum
        dvh = np.flip(dvh,0)  # Flip back to ascending order
        dvh = dvh / np.sum(hist) * 100.0  # Normalize to percentage
        self.bin_dose = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # dose at midpoints of bins
        self.dvh = dvh
        self.DMin = dose_mask.min()
        self.DMax = dose_mask.max()
        self.DMean = dose_mask.mean()
    
    

    def computeDx(self, x: float) -> float:
        diff_array = np.abs(self.dvh - x)
        index = np.argmin(diff_array)
        return self.bin_dose[index]
    
    def computeVx(self, x: float) -> float:
        diff_array = np.abs(self.bin_dose - x)
        index = np.argmin(diff_array)
        return self.dvh[index]
    

    

    