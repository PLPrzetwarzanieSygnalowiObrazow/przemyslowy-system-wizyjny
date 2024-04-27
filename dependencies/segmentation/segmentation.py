import numpy
from skimage import measure
from dataclasses import dataclass


@dataclass
class Segmentation:
    
    @staticmethod
    def label(frame: numpy.ndarray) -> numpy.ndarray:
        return (
            measure.label(frame)
        )

    @staticmethod
    def regionprops(frame: numpy.ndarray) -> numpy.ndarray:
        return (
            measure.regionprops(frame)
        )
