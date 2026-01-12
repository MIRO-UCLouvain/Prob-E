import numpy as np

class PatientData:
    def __init__(self, ctImage, doseImage, targetMask):
        self.ctImage = ctImage
        self.doseImage = doseImage
        self.targetMask = targetMask

    def get_ct_image(self):
        return self.ctImage
    
    def get_dose_image(self):
        return self.doseImage
    
    def get_target_mask(self):
        return self.targetMask
    
    def set_ct_image(self, ctImage):
        self.ctImage = ctImage
    
    def set_dose_image(self, doseImage):
        self.doseImage = doseImage

    def set_target_mask(self, targetMask):
        self.targetMask = targetMask

        