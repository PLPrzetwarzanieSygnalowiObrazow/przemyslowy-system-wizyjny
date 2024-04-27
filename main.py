from skimage import filters, feature, segmentation, morphology, color
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import copy

from dependencies.video.video import Video


def main():
    video = Video(path="./assets/conveyor-jewelry.mp4")
    video.get_frame()


if __name__ == "__main__":
    main()
