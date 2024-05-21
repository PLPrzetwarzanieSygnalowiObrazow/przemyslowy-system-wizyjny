import numpy
import cv2
import argparse
from dependencies.descriptions import *
from dependencies.video import Video
from dependencies.filter import Filter
from dependencies.segmentation import Segmentation
from dependencies.draw import Draw
from dependencies.blobDetectorInit import RINGS_DETECTOR, EARINGS_DETECTOR, NECKLACES_DETECTOR
from dependencies.objectTracker import ObjectTracker

VIDEO_FILE_PATH = "./assets/nagranie_v4_cut.mp4"
STARTING_FRAME_NO = 0


def main():

    while (org_frame := video.get_frame()) is not None:

        transformedFrames = transformFrame(org_frame)

        detectedObjects = detectObjects(transformedFrames)

        frame_to_display = countObjects(detectedObjects, org_frame)

        video.show_frame(frame_to_display)

    tracker.printTrackingReport()

    return None


def transformFrame(
    frame: numpy.ndarray
) -> tuple[
    numpy.ndarray,
    numpy.ndarray
]:
    '''
        Returns 2 frames in tuple.
        First one is dedicated for detecting the rings,
        the second one for necklaces and earings
    '''
    # Zamiana klatki na odcienie szarości
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rozmazanie klatki
    gaussian_frame = Filter.gauss(gray_frame)

    # Wykrywanie krawędzi
    canny_frame = Filter.canny(gaussian_frame)

    # Domknięcie krawędzi
    rings_frame = Filter.closing(canny_frame, 2)

    # Znalezienie krawędzi
    contours = Segmentation.findContours(rings_frame)

    # Wypełnienie znalezionych krawędzi
    ear_neck_frame = Draw.contourFill(gray_frame, contours)

    return (rings_frame, ear_neck_frame)


def detectObjects(
    framesForDetection: tuple[
        numpy.ndarray,
        numpy.ndarray
    ]
) -> tuple[
    tuple[cv2.KeyPoint],
    tuple[cv2.KeyPoint],
    tuple[cv2.KeyPoint]
]:
    '''
        Returns 3 tuples of key points for different types of objects:
            - rings,
            - earings,
            - necklaces.
    '''
    # Rozpakowanie tuple przygotowanych ramek
    rings_frame = framesForDetection[0]
    ear_neck_frame = framesForDetection[1]

    # Znalezienie pierścionków
    rings_KP = RINGS_DETECTOR.detect_objects(rings_frame)

    # Znalezienie kolczyków
    earings_KP = EARINGS_DETECTOR.detect_objects(ear_neck_frame)

    # Znalezienie naszyjników
    necklaces_KP = NECKLACES_DETECTOR.detect_objects(ear_neck_frame)

    return (rings_KP, earings_KP, necklaces_KP)


def countObjects(
    detectedObjects: tuple[
        tuple[cv2.KeyPoint],
        tuple[cv2.KeyPoint],
        tuple[cv2.KeyPoint]
    ],
    frame_to_mark_objects: numpy.ndarray
) -> numpy.ndarray:
    '''
        Returns frame passed in, with marked objects which were found
    '''
    # Rozpakowanie tuple przygotowanych key points odpowiadającym konkretnemu typowi obiektu
    rings_KP = detectedObjects[0]
    earings_KP = detectedObjects[1]
    necklaces_KP = detectedObjects[2]

    # Identyfikacja obiektów i zaznaczenie ich na ramce
    return tracker.trackObjects(
        rings_key_points=rings_KP,
        necklaces_key_points=necklaces_KP,
        earings_key_points=earings_KP,
        frame_to_draw=frame_to_mark_objects
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument(
        "-p",
        "--video_file_path",
        default=VIDEO_FILE_PATH,
        type=str,
        help=VIDEO_FILE_PATH_HELPER
    )
    parser.add_argument(
        "-f",
        "--start_frame_number",
        default=STARTING_FRAME_NO,
        type=int,
        help=START_FRAME_NUMBER_HELPER
    )
    args = parser.parse_args()
    video = Video(
        path=args.video_file_path,
        frame_no=args.start_frame_number
    )

    tracker = ObjectTracker()
    main()
