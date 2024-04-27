import cv2
from dataclasses import dataclass, field
from typing import Final
import numpy


@dataclass
class Video:
    path: str = field(default=None)
    width: int = field(default=640)
    height: int = field(default=480)
    frame_no: int = field(default=0, init=False)
    ESC_KEY: Final[int] = 27

    capture: cv2.VideoCapture = field(default=None, init=False)
    current_frame: numpy.ndarray = field(default=None, init=False)
    # current_frame_no:

    def __post_init__(self):
        if self.path is None:
            raise Exception("Path not defined")

        self.__open()

    def __open(self) -> None:
        self.capture = cv2.VideoCapture(self.path)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        while not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.path)
            print("Waiting for video")
            key = cv2.waitKey(2000)
            if key == self.ESC_KEY:
                return

    def get_frame(self) -> numpy.ndarray:
        flag, frame = self.capture.read()

        if flag:
            frame_no = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            cv2.imshow("Obraz", frame)
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            print("Ramka nie jest gotowa")
            cv2.waitKey(100)

        if cv2.waitKey(10) == self.ESC_KEY:
            cv2.destroyAllWindows()
            # self.capture.release()

        return frame

    def show_frame(self) -> None:
        cv2.imshow("Frame", self.current_frame)
