import cv2
import numpy
from skimage import filters, feature, segmentation, morphology, color, measure
from dataclasses import dataclass, field
from typing import Final


@dataclass
class BlobDetector:

    filter_by_area: bool = field(default=True, init=True)
    min_area: int = field(default=400, init=True)
    max_area: int = field(default=40000, init=True)
    filter_by_circularity: bool = field(default=True, init=True)
    min_circularity: float = field(default=0.05, init=True)
    filter_by_convexity: bool = field(default=False, init=True)
    min_convexity: int = field(default=0.1, init=True)
    filter_by_inertia: bool = field(default=False, init=True)
    min_inertia_ratio: float = field(default=0.2, init=True)

    detector_params: cv2.SimpleBlobDetector = field(
        default=cv2.SimpleBlobDetector_Params(),
        init=False
    )
    blob_detector: cv2.SimpleBlobDetector = field(default=None, init=False)

    def __post_init__(self):
        self.detector_params.filterByArea = self.filter_by_area
        self.detector_params.minArea = self.min_area
        self.detector_params.maxArea = self.max_area
        self.detector_params.filterByCircularity = self.filter_by_circularity
        self.detector_params.minCircularity = self.min_circularity
        self.detector_params.filterByConvexity = self.filter_by_convexity
        self.detector_params.minConvexity = self.min_convexity
        self.detector_params.filterByInertia = self.filter_by_inertia
        self.detector_params.minInertiaRatio = self.min_inertia_ratio
        
        self.blob_detector = cv2.SimpleBlobDetector_create(
            self.detector_params
        )

    def detect_objects(self, frame: numpy.ndarray) -> tuple:
        return (
            self.blob_detector.detect(frame)
        )
