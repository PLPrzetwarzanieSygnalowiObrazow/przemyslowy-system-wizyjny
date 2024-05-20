from skimage import filters, feature, segmentation, morphology, color
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import copy

from dependencies.video import Video
from dependencies.filter import Filter
from dependencies.blobDetector import BlobDetector
from dependencies.draw import Draw
from dependencies.segmentation import Segmentation
from dependencies.objectTracker import ObjectTracker

VIDEO_FILE_PATH = "./assets/nagranie_v4_cut.mp4"


def main():
    video = Video(path=VIDEO_FILE_PATH, frame_no=550)

    EARINGS_DETECTOR = BlobDetector(
        filter_by_color=True,
        blob_color=0,
        filter_by_area=True,
        min_area=800,
        max_area=6_200,
        filter_by_circularity=True,
        min_circularity=0.25,
        max_circularity=0.65,
        filter_by_convexity=True,
        min_convexity=0.15,
        max_convexity=0.80,
        filter_by_inertia=True,
        min_inertia_ratio=0.02,
        max_inertia_ratio=0.60,
    )
    RINGS_DETECTOR = BlobDetector(
        filter_by_color=True,
        blob_color=0,
        filter_by_area=True,
        min_area=7000/(1920/video.width),
        max_area=12000/(1920/video.width),
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

    NECKLACES_DETECTOR = BlobDetector(
        filter_by_color=True,
        blob_color=0,
        filter_by_area=True,
        min_area=100_000,
        max_area=900_000,
        filter_by_circularity=True,
        min_circularity=0.25,
        max_circularity=0.9,
        filter_by_convexity=False,
        min_convexity=0.1,
        max_convexity=1.0,
        filter_by_inertia=False,
        min_inertia_ratio=0.4,
        max_inertia_ratio=1,
    )
    tracker = ObjectTracker()

    while (org_frame := video.get_frame()) is not None:
        # Zamiana klatki na odcienie szarości
        gray_frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)

        # Rozmazanie klatki
        gaussian_frame = Filter.gauss(gray_frame)

        # Wykrywanie krawędzi
        canny_frame = Filter.canny(gaussian_frame)

        # Domknięcie krawędzi
        closed_frame = Filter.closing(canny_frame, 2)

        # Znalezienie krawędzi
        contours = Segmentation.findContours(closed_frame)

        # Wypełnienie znalezionych krawędzi
        contours_filled = Draw.contourFill(gray_frame, contours)

        # Zamknięcie krawędzi (tylko do pierścionków)
        # closed_frame = Filter.closing(canny_frame, 4)
        # print(canny_frame.shape)
        # Znalezienie pierścionków
        # rings_key_points = RINGS_DETECTOR.detect_objects(closed_frame)

        # Znalezienie kolczyków
        earings_key_points = EARINGS_DETECTOR.detect_objects(contours_filled)

        # Znalezienie naszyjników
        necklaces_key_points = NECKLACES_DETECTOR.detect_objects(
            contours_filled)

        # tracker.trackObjects(rings_key_points=rings_key_points)
        # Połączenie obrazu głównego z punktami
        # Wyszukiwanie pierścieni działa najlepiej
        # result = Draw.keyPoints(org_frame, rings_key_points, Draw.COLOR_RED)
        result = Draw.keyPoints(
            contours_filled, earings_key_points, Draw.COLOR_GREEN)
        result = Draw.keyPoints(result, necklaces_key_points, Draw.COLOR_BLUE)

        # result = Draw.rectangle(
        #     result, [[(200, 0), (300, 300)]], color=Draw.COLOR_RED)

        video.show_frame(result)


if __name__ == "__main__":
    main()
