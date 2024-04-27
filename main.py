from skimage import filters, feature, segmentation, morphology, color
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import time
import copy


ESC_KEY = 27


def main():
    #device = 0 # łączenie z kamerką
    device = './assets/conveyor-jewelry.mp4'
    cap = cv2.VideoCapture(device)
    frame_no = 0    
    tracked = {}
    archive = {}
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while not cap.isOpened():
        cap = cv2.VideoCapture(device)
        print('Czekam na wideo')
        k = cv2.waitKey(2000)
        if k == ESC_KEY:
            return
        
    keep = True
    while keep:
        flag, frame = cap.read()
        if flag:
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cv2.imshow('Obraz', frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
            print('Ramka nie jest gotowa')
            cv2.waitKey(100)


        if cv2.waitKey(10) == ESC_KEY:
            keep = False
            cv2.destroyAllWindows()
            cap.release()
            
            break
        
        
if __name__ == '__main__':
    main()
    