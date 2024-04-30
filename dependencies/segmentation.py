import numpy
import cv2
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

    @staticmethod
    def findContours(frame: numpy.ndarray, mode: int = cv2.RETR_CCOMP, method: int = cv2.CHAIN_APPROX_SIMPLE):
        return (
            cv2.findContours(
                image=frame,
                mode=mode,
                method=method
            )
        )[0]
