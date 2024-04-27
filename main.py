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

VIDEO_FILE_PATH = "./assets/conveyor-jewelry.mp4"

def main():
    video = Video(path=VIDEO_FILE_PATH)
    
    while not video.is_ended():
        
        fr = video.get_frame()
        video.show_frame(fr)
    


if __name__ == "__main__":
    main()
