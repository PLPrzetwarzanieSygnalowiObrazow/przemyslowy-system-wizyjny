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

VIDEO_FILE_PATH = "./assets/conveyor-jewelry.mp4"


def main():
    video = Video(path=VIDEO_FILE_PATH)

    bd = BlobDetector()

    while not video.is_ended():
        org_frame = video.get_frame()
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
        filtered_frame = Filter.gauss(frame)
        filtered_frame = Filter.canny(filtered_frame)

        #filtered_frame = Filter.clear_border(filtered_frame, 1)
        filtered_frame = Filter.closing(filtered_frame, 3)
        
        closing_mask = np.ones((12, 12), np.uint8)
        filtered_frame = cv2.morphologyEx(
            src=filtered_frame,
            op=cv2.MORPH_CLOSE,
            kernel=closing_mask
        )

        result = np.zeros_like(org_frame)
        red_color = (255, 0, 0)
        result[:] = red_color
        # filtered_frame = Filter.closing(filtered_frame, 3)
        # filtered_frame = Filter.closing(filtered_frame, 6)
        contours, _ = cv2.findContours(
            image=filtered_frame,
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        for i, contour in enumerate(contours):
            color = (0, 255, 0)

            # wypelnianie konturu w odpowiednim kolorze
            cv2.drawContours(result, [contour], -1, color, -1)
        # filtered_frame = Filter.remove_small_objects(filtered_frame, 250)

        keyPoints = bd.detect_objects(result)

        video.show_frame(
            cv2.drawKeypoints(
                image=result,
                keypoints=keyPoints,
                outImage=np.array([]),
                color=(0, 0, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        )
        
        video.show_frame(result)


if __name__ == "__main__":
    main()
