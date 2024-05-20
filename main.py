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

from dependencies.draw import Draw
from dependencies.segmentation import Segmentation
from dependencies.blobDetectorInit import *
from dependencies.objectTracker import ObjectTracker


VIDEO_FILE_PATH = "./assets/nagranie_v4_cut.mp4"


def main():
    video = Video(path=VIDEO_FILE_PATH, frame_no=400)

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
