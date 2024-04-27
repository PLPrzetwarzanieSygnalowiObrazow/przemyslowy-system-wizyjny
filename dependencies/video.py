import cv2
from dataclasses import dataclass, field
from typing import Final
import numpy


@dataclass
class Video:
    
    WINDOW_TITLE: Final[str] = "Frame"
    MSG_VIDEO_ENDED: Final[str] = "Video ended"

    DELAY_FOR_OPEN_FILE: Final[int] = 2000
    DELAY_BETWEEN_FRAMES: Final[int] = 1
    RETRY_LIMIT_GET_FRAME: Final[int] = 3
    RETRY_LIMIT_OPEN: Final[int] = 3

    
    path: str = field(default=None)
    width: int = field(default=640)
    height: int = field(default=480)

    frame_no: int = field(default=0, init=False)
    frame_flag: bool = field(default=True, init=False)
    capture: cv2.VideoCapture = field(default=None, init=False)
    current_frame: numpy.ndarray = field(default=None, init=False)
    
    def __post_init__(self):
        if self.path is None:
            raise Exception("Path not defined")

        self.__open()

    def __del__(self):
        cv2.destroyAllWindows()
        self.capture.release()

    def __open(self) -> None:
        
        # Init retry counter
        retry_counter = 0

        # Attempt/retry to open file, until number of retries is not exceeded
        while ((retry_counter := retry_counter + 1) <= self.RETRY_LIMIT_OPEN):
            
            # Open video file
            self.capture = cv2.VideoCapture(self.path)            
            cv2.waitKey(self.DELAY_FOR_OPEN_FILE)
            
            # Check if file was opened
            if self.capture.isOpened():
                # Set video resolution
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                break
        
        # execute if loop was not exited with break
        # which means that file was not opened successfully
        else:
            raise Exception("Could not open video file")

    def pause(self, delay_counter: int = 1) -> None:
        cv2.waitKey(pow(self.DELAY_BETWEEN_FRAMES, delay_counter))

    def get_frame(self) -> numpy.ndarray:
        
        # Init retry counter
        retry_counter = 0
        
        # Attempt/retry to get a frame. There is a chance that it might failed previous time
        # so we try RETRY_LIMIT_GET_FRAME times to read the frame.
        while (retry_counter := retry_counter + 1) <= self.RETRY_LIMIT_GET_FRAME:
            self.frame_flag, self.current_frame = self.capture.read()
            if self.frame_flag:
                self.frame_no = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                break
            self.pause(delay_counter=retry_counter)
        
        # Execute if loop was not exited by break
        else:
            print(self.MSG_VIDEO_ENDED)
            return None
        
        return self.current_frame
    
    def get_gray_frame(self) -> numpy.ndarray:
        self.current_frame = cv2.cvtColor(
            self.get_frame(),
            cv2.COLOR_BGR2GRAY
        )
        
        return self.current_frame

    def show_frame(self, frame: numpy.ndarray = None) -> None:
        if self.frame_flag is False:
            return

        if frame is None:
            self.get_frame()
            cv2.imshow(self.WINDOW_TITLE, self.current_frame)
        else:
            cv2.imshow(self.WINDOW_TITLE, frame)

        self.pause()

    def is_ended(self) -> bool:
        return self.frame_flag is False
