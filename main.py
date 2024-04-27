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

VIDEO_FILE_PATH = "./assets/nagranie_v4_cut.mp4"


def main():
    video = Video(path=VIDEO_FILE_PATH, frame_no=550)

    bd = BlobDetector()

    while not video.is_ended():
        org_frame = video.get_frame()
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
        filtered_frame = Filter.gauss(frame)
        
        filtered_frame = Filter.canny(filtered_frame)
        
        # usunięcie artefaktów na brzegach obrazu
        cleared_edges = segmentation.clear_border(
            labels=filtered_frame,
            buffer_size=0
        )
        
        # filtr morfologiczny
        closing_mask = np.ones((4, 4), np.uint8)
        closed_edges = cv2.morphologyEx(
            src=cleared_edges,
            op=cv2.MORPH_CLOSE,
            kernel=closing_mask
        )
        contours, hierarchy = cv2.findContours(
            image=closed_edges,
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        result = np.zeros_like(org_frame)
        red_color = (0, 0, 255)
        result[:] = red_color

        video.show_frame(
            Draw.contourFill(
                result,
                contours,
                Draw.COLOR_GREEN
            )
        )


if __name__ == "__main__":
    main()
