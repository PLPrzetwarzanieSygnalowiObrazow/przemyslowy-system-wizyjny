from dataclasses import dataclass, field
import cv2
import math

X = 0
Y = 1


@dataclass
class JewelryObject:
    '''
        Parent class to define jewelry object.
        Configurable attributes are written in Capital letters only.
        Inheritance classes defines following attributes:
            - X_AXIS_MOVEMENT_ERROR <- movement threshold in X axis (in pixels),
                which will take up any imperfections in conveyor belt movement.
            - Y_AXIS_MOVEMENT_ERROR <- movement threshold in Y axis (in pixels),
                which will take up any imperfections in conveyor belt movement.
            - X_AXIS_MOVEMENT_PER_FRAME <- represents movement of objects in the frame horizontally,
                it is a result of moving conveyor belt. Value is a number of pixels.
            - Y_AXIS_MOVEMENT_PER_FRAME <- represents movement of objects in the frame vertically,
                allows to take up scenarios where belt is not moving ideally horizontal. Value is a number of pixels.
            - MARK_AS_INVISIBLE_AFTER_X_COORDINATE <- threshold where objects might disappear from frame.
                Value is a number of pixels.
            - MARK_AS_INVISIBLE_AFTER_MISSING_ON_FRAMES <- represents the number of frames,
                where object must be missing, to be considered as not visible.

        WARNING: To mark object as not visible ALL of the following criteria values must be met:
            - MARK_AS_INVISIBLE_AFTER_X_COORDINATE
            - MARK_AS_INVISIBLE_AFTER_MISSING_ON_FRAMES
    '''

    OBJECT_NAME: str = field(init=True, default="Obiekt biżuteryjny")

    X_AXIS_MOVEMENT_ERROR: int = field(init=False, default=None)
    Y_AXIS_MOVEMENT_ERROR: int = field(init=False, default=None)

    X_AXIS_MOVEMENT_PER_FRAME: int = field(init=False, default=None)
    Y_AXIS_MOVEMENT_PER_FRAME: int = field(init=False, default=None)

    MARK_AS_INVISIBLE_AFTER_X_COORDINATE: int = field(init=False, default=None)

    MARK_AS_INVISIBLE_AFTER_MISSING_ON_FRAMES: int = field(
        init=False,
        default=10
    )

    positions: list[cv2.KeyPoint] = field(init=False, default_factory=list)

    __missing_on_frames: int = field(init=False, default=0)
    __found_on_frames: int = field(init=False, default=1)
    __appended: bool = field(init=False, default=True)
    __visible: bool = field(init=False, default=True)

    def __post_init__(self):
        print(f"New {self.OBJECT_NAME} found")

    def isVisible(self) -> bool:
        return self.__visible

    def getFoundOnFrames(self) -> int:
        return self.__found_on_frames

    def getMissingOnFrames(self) -> int:
        return self.__missing_on_frames

    def appendPositions(self, key_point: cv2.KeyPoint):
        self.positions.append(key_point)
        self.__appended = True
        self.__missing_on_frames = 0
        self.__found_on_frames = self.__found_on_frames + 1

    def resetAppendFlag(self):
        self.__appended = False

    def incrementMissingOnFrames(self):

        # if object is not visible return, as no further checks are needed
        if not self.__visible:
            return None

        # check if the number of frames where object was missing exceeded the threshold AND
        # if the last object's position was near end of frame
        if (
            (self.__missing_on_frames >= self.MARK_AS_INVISIBLE_AFTER_MISSING_ON_FRAMES) and
            (
                self.positions[-1].pt[X] >= self.MARK_AS_INVISIBLE_AFTER_X_COORDINATE
            )
        ):
            # mark object as not visible, print message in console and return
            self.__visible = False
            print(f"{self.OBJECT_NAME} marked as invisible")
            return None

        # if object was not appended and no other conditions were met increment number of frames,
        # where object was missing
        if not self.__appended:
            self.__missing_on_frames = self.__missing_on_frames + 1

        return None

    def calculateDistance(self, key_point: cv2.KeyPoint = None) -> tuple[float, float]:

        # get last position and possible movement comparing to the last time when object was visible
        lastPosition = self.getLastPosition()
        movement_estimation = self.getAcceptableMovement()

        # calculate absolute position changes between last known position and the one provided to method,
        # including estimated movement
        x_distance = abs(
            key_point.pt[X] - (lastPosition.pt[X] + movement_estimation[X])
        )
        y_distance = abs(
            key_point.pt[Y] - (lastPosition.pt[Y] + movement_estimation[Y])
        )

        return (x_distance, y_distance)

    def getAcceptableMovement(self) -> tuple[int, int]:

        # return possible movement multiplied by the number of frames where object was missing.
        # even if the object was not identified it was moving on conveyor belt
        return (
            self.__missing_on_frames * self.X_AXIS_MOVEMENT_PER_FRAME,
            self.__missing_on_frames * self.Y_AXIS_MOVEMENT_PER_FRAME
        )

    def getLastPosition(self) -> cv2.KeyPoint:
        return self.positions[-1]


@dataclass
class Ring (JewelryObject):
    OBJECT_NAME: str = "Pierścionek"

    X_AXIS_MOVEMENT_PER_FRAME: int = 4
    Y_AXIS_MOVEMENT_PER_FRAME: int = 1

    X_AXIS_MOVEMENT_ERROR: int = 10
    Y_AXIS_MOVEMENT_ERROR: int = 10

    MARK_AS_INVISIBLE_AFTER_X_COORDINATE: int = 1_000


@dataclass
class Necklace (JewelryObject):
    OBJECT_NAME: str = "Naszyjnik"

    X_AXIS_MOVEMENT_PER_FRAME: int = 0
    Y_AXIS_MOVEMENT_PER_FRAME: int = 0

    X_AXIS_MOVEMENT_ERROR: int = 38
    Y_AXIS_MOVEMENT_ERROR: int = 38

    MARK_AS_INVISIBLE_AFTER_X_COORDINATE: int = 800


@dataclass
class Earings (JewelryObject):
    OBJECT_NAME: str = "Kolczyki"

    X_AXIS_MOVEMENT_PER_FRAME: int = 0
    Y_AXIS_MOVEMENT_PER_FRAME: int = 0

    X_AXIS_MOVEMENT_ERROR: int = 50
    Y_AXIS_MOVEMENT_ERROR: int = 50

    DISTANCE_BETWEEN_EARINGS: int = 100

    MARK_AS_INVISIBLE_AFTER_X_COORDINATE: int = 1_000
    MARK_AS_INVISIBLE_AFTER_MISSING_ON_FRAMES: int = 40

    @staticmethod
    def groupEaringsIntoPairs(keyPoints: tuple[cv2.KeyPoint]) -> tuple[cv2.KeyPoint]:

        output: set[cv2.KeyPoint] = set()
        distances: list[tuple[float, cv2.KeyPoint, cv2.KeyPoint]] = []
        paired_KPs: set[int] = set()
        for i in range(0, len(keyPoints)):

            for j in range(0, len(keyPoints)):
                if i == j:
                    continue
                distances.append(
                    (
                        Earings.__get_objects_distance(
                            keyPoints[i],
                            keyPoints[j]
                        ),
                        i,
                        j
                    )
                )

        distances.sort()

        for dist, i, j in distances:
            if i not in paired_KPs and j not in paired_KPs and dist < Earings.DISTANCE_BETWEEN_EARINGS:
                paired_KPs.add(i)
                paired_KPs.add(j)
                output.add(
                    Earings.__merge_key_points(
                        keyPoints[i],
                        keyPoints[j]
                    )
                )

        return tuple(output)

    @staticmethod
    def __merge_key_points(kp1: cv2.KeyPoint, kp2: cv2.KeyPoint) -> cv2.KeyPoint:
        return cv2.KeyPoint(
            (kp1.pt[X] + kp2.pt[X]) / 2,
            (kp1.pt[Y] + kp2.pt[Y]) / 2,
            (kp1.size + kp2.size) / 2
        )

    @staticmethod
    def __get_objects_distance(KP_1: cv2.KeyPoint, KP_2: cv2.KeyPoint) -> float:
        return math.sqrt(
            (KP_1.pt[X] - KP_2.pt[X]) ** 2 +
            (KP_1.pt[Y] - KP_2.pt[Y]) ** 2
        )
