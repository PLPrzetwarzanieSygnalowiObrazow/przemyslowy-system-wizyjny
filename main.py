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

VIDEO_FILE_PATH = "./assets/conveyor-jewelry.mp4"


def main():
    video = Video(path=VIDEO_FILE_PATH)

    bd = BlobDetector()

    while not video.is_ended():
        org_frame = video.get_frame()
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
        filtered_frame = Filter.gauss(frame)
        filtered_frame = Filter.canny(filtered_frame)

        # filtered_frame = Filter.clear_border(filtered_frame, 1)
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
        
        result = Draw.contourFill(result,contours, Draw.COLOR_GREEN)
        # filtered_frame = Filter.remove_small_objects(filtered_frame, 250)

        keyPoints = bd.detect_objects(result)

        video.show_frame(
            Draw.keyPoints(
                result,
                keyPoints,
                Draw.COLOR_RED
            )
        )

        # video.show_frame(
        #     Draw.rectangle(
        #         result,
        #         [[(0,0),(300,300)],[(400,400),(600,600)]],
        #         Draw.COLOR_RED
        #     )
        # )


if __name__ == "__main__":
    main()
