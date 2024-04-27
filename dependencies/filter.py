import cv2
import numpy
from skimage import filters, feature, segmentation, morphology, color, measure
from dataclasses import dataclass, field
from typing import Final


@dataclass
class Filter:

    GAUSS_SIGMA_X: Final[int] = 0
    GAUSS_SIGMA_Y: Final[int] = 0
    GAUSS_K_SIZE: Final[float] = (5, 5)

    CANNY_THR_1: Final[int] = 60
    CANNY_THR_2: Final[int] = 255

    @staticmethod
    def gauss(frame: numpy.ndarray) -> numpy.ndarray:
        return (
            cv2.GaussianBlur(
                src=frame,
                ksize=Filter.GAUSS_K_SIZE,
                sigmaX=Filter.GAUSS_SIGMA_X,
                sigmaY=Filter.GAUSS_SIGMA_Y
            )
        )

    @staticmethod
    def canny(frame: numpy.ndarray) -> numpy.ndarray:
        return (
            cv2.Canny(
                image=frame,
                threshold1=Filter.CANNY_THR_1,
                threshold2=Filter.CANNY_THR_2
            )
        )

    @staticmethod
    def clear_border(frame: numpy.ndarray, buffer_size: int) -> numpy.ndarray:
        return (
            segmentation.clear_border(
                labels=frame,
                buffer_size=buffer_size
            )
        )

    @staticmethod
    def closing(frame: numpy.ndarray, disk_radius: int) -> numpy.ndarray:
        return (
            morphology.closing(
                image=frame,
                footprint=morphology.disk(disk_radius)
            )
        )

    @staticmethod
    def remove_small_objects(frame: numpy.ndarray, min_size: int) -> numpy.ndarray:
        return (
            morphology.remove_small_objects(
                ar=frame,
                min_size=min_size
            )
        )
