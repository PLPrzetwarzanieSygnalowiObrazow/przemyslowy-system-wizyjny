from skimage import filters, feature, segmentation, morphology, color
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import copy
from scipy import ndimage as ndi
from skimage import measure
from dependencies.video import Video
from dependencies.filter import Filter
from dependencies.blobDetector import BlobDetector
from dependencies.draw import Draw

VIDEO_FILE_PATH = "./assets/nagranie_v4_cut.mp4"
EARINGS_DETECTOR = BlobDetector(
    filter_by_color=True,
    blob_color=0,
    filter_by_area=True,
    min_area=100,
    max_area=8000,
    filter_by_circularity=True,
    min_circularity=0.3,
    max_circularity=0.8,
    filter_by_convexity=True,
    min_convexity=0.01,
    max_convexity=0.8,
    filter_by_inertia=True,
    min_inertia_ratio=0.01,
    max_inertia_ratio=0.8,
)
RINGS_DETECTOR = BlobDetector(
    filter_by_color=True,
    blob_color=0,
    filter_by_area=True,
    min_area=7000,
    max_area=12000,
    filter_by_circularity=True,
    min_circularity=0.6,
    max_circularity=1,
    filter_by_convexity=True,
    min_convexity=0.6,
    max_convexity=1,
    filter_by_inertia=True,
    min_inertia_ratio=0.5,
    max_inertia_ratio=1,
)
NECKLES_DETECTOR = BlobDetector(
    filter_by_color=True,
    blob_color=0,
    filter_by_area=True,
    min_area=60000,
    max_area=300000,
    filter_by_circularity=False,
    min_circularity=0.3,
    max_circularity=1,
    filter_by_convexity=False,
    min_convexity=0.1,
    max_convexity=0.5,
    filter_by_inertia=False,
    min_inertia_ratio=0.4,
    max_inertia_ratio=1,
)


def main():
    video = Video(path=VIDEO_FILE_PATH, frame_no=550)

    while not video.is_ended():
        # Pobranie pojedynczej klatki
        org_frame = video.get_frame()
        if org_frame is None:
            break

        # Zamiana klatki na odcienie szarości
        gray_frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)

        # Rozmazanie klatki
        gaussian_frame = Filter.gauss(gray_frame)

        # Wykrywanie krawędzi
        canny_frame = Filter.canny(gaussian_frame)

        # Domknięcie krawędzi
        closed_frame = Filter.closing(canny_frame, 2)
        closed_frame = Filter.closing(canny_frame, 4)

        # Znalezienie pierścionków
        rings_key_points = RINGS_DETECTOR.detect_objects(closed_frame)

        # Znalezienie kolczyków
        earings_key_points = EARINGS_DETECTOR.detect_objects(closed_frame)

        # Znalezienie naszyjników
        neckles_key_points = NECKLES_DETECTOR.detect_objects(closed_frame)

        # Połączenie obrazu głównego z punktami
        # Wyszukiwanie pierścieni działa najlepiej
        result = Draw.keyPoints(org_frame, rings_key_points, Draw.COLOR_RED)
        result = Draw.keyPoints(result, earings_key_points, Draw.COLOR_GREEN)
        result = Draw.keyPoints(result, neckles_key_points, Draw.COLOR_BLUE)

        video.show_frame(result)


if __name__ == "__main__":
    main()
