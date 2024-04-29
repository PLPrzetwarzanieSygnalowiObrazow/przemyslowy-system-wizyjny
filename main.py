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


def mainContours():
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


def mainThresholds():
    video = Video(path=VIDEO_FILE_PATH, frame_no=550)

    bd_earings = BlobDetector(
        filter_by_area=True,
        min_area=1000,
        max_area=10800,
        filter_by_circularity=True,
        min_circularity=0.3,
        max_circularity=0.5,
        filter_by_convexity=True,
        min_convexity=0.86,
        max_convexity=0.9,
        filter_by_inertia=True,
        min_inertia_ratio=0.1,
        max_inertia_ratio=0.9
    )

    while not video.is_ended():
        org_frame = video.get_frame()
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
        thr = filters.threshold_minimum(frame)

        thresh_for_frame = (frame >= thr)

        # fr = ndi.binary_fill_holes(thresh_for_frame)
        # brain_l, lb_num = measure.label(fr, return_num=True)
        # num_of_regions = np.max(brain_l)
        # regionProps = measure.regionprops(brain_l)
        # areas = [prop.area for prop in regionProps]
        # max_index = np.argmax(areas)
        # seg_brain = (brain_l==regionProps[max_index].label)
        # brain_size = int(areas[max_index])

        frame_converted = np.uint8(thresh_for_frame) * 255
        filtered_frame = cv2.GaussianBlur(
            src=frame_converted,
            ksize=(1, 1),
            sigmaX=0,
            sigmaY=0
        )

        filtered_frame = cv2.Canny(
            image=filtered_frame,
            threshold1=0,
            threshold2=255
        )

        closing_mask = np.ones((5, 1), np.uint8)
        closed_edges = cv2.morphologyEx(
            src=filtered_frame,
            op=cv2.MORPH_CLOSE,
            kernel=closing_mask
        )
        contours, hierarchy = cv2.findContours(
            image=closed_edges,
            mode=cv2.RETR_CCOMP,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        keyPoints = bd_earings.detect_objects(Draw.contourFill(
                frame,
                contours,
                Draw.COLOR_GREEN
            ))

        # video.show_frame(
        #     filtered_frame
        # )
        video.show_frame(
            Draw.keyPoints(
                frame,
                keyPoints,
                Draw.COLOR_GREEN
            )
        )


if __name__ == "__main__":
    mainThresholds()
