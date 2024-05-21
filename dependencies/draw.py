import cv2
import numpy
from dataclasses import dataclass


@dataclass
class Draw:

    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)

    RECTANGLE_THICKNESS = 5

    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 1
    TEXT_THICKNESS = 2
    TEXT_LINE_TYPE = cv2.LINE_AA

    CONTOUR_THICKNESS = -1
    CONTOUR_IDX = -1

    X = 0
    Y = 1

    @staticmethod
    def keyPoints(frame: numpy.ndarray, keyPoints: tuple, color: tuple[int, int, int]) -> numpy.ndarray:
        return (
            cv2.drawKeypoints(
                image=frame,
                keypoints=keyPoints,
                outImage=numpy.array([]),
                color=color,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        )

    @staticmethod
    def rectangle(frame: numpy.ndarray, points: list[list[tuple[int, int]]], color: tuple[int, int, int]) -> numpy.ndarray:
        for p1, p2 in points:
            cv2.rectangle(
                img=frame,
                pt1=p1,
                pt2=p2,
                color=color,
                thickness=Draw.RECTANGLE_THICKNESS
            )

        return frame

    @staticmethod
    def text(frame: numpy.ndarray, text: str, color: tuple[int, int, int], point: tuple[int, int]):
        cv2.putText(
            img=frame,
            text=text,
            org=point,
            fontFace=Draw.TEXT_FONT,
            fontScale=Draw.TEXT_FONT,
            color=color,
            thickness=Draw.TEXT_THICKNESS,
            lineType=Draw.TEXT_LINE_TYPE
        )

        return frame

    @staticmethod
    def contourFill(frame: numpy.ndarray, contours: list[tuple], color: tuple[int, int, int] = (0, 0, 0)):
        for i, contour in enumerate(contours):
            cv2.drawContours(
                image=frame,
                contours=[contour],
                contourIdx=Draw.CONTOUR_IDX,
                color=color,
                thickness=Draw.CONTOUR_THICKNESS
            )

        return frame

    @staticmethod
    def contour(frame: numpy.ndarray, contours: list[tuple], color: tuple[int, int, int]):
        cv2.drawContours(
            image=frame,
            contours=contours,
            contourIdx=Draw.CONTOUR_IDX,
            color=color,
            thickness=Draw.CONTOUR_THICKNESS
        )

        return frame
