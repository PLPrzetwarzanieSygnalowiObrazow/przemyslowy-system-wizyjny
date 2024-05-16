from dataclasses import dataclass, field
import cv2

X = 0
Y = 1


@dataclass
class JewelryObject:

    positions: list[cv2.KeyPoint] = field(init=False, default_factory=list)
    visible: bool = field(init=False, default=True)
    X_AXIS_MOVEMENT_PER_FRAME: int = field(init=False, default=25)
    Y_AXIS_MOVEMENT_PER_FRAME: int = field(init=False, default=10)
    __missing_on_frames: int = field(init=False, default=3)

    def getID(self) -> int:
        return self.ID

    def appendPositions(self, key_point: cv2.KeyPoint):
        self.positions.append(key_point)

    def calculateDistance(self, key_point: cv2.KeyPoint = None):
        lastPosition = self.positions[-1]
        x_distance = abs(key_point.pt[X] - lastPosition.pt[X])
        y_distance = abs(key_point.pt[Y] - lastPosition.pt[Y])
        return (x_distance, y_distance)

    def getAcceptableMovement(self) -> tuple[int, int]:
        return (
            self.__missing_on_frames * self.X_AXIS_MOVEMENT_PER_FRAME,
            self.__missing_on_frames * self.Y_AXIS_MOVEMENT_PER_FRAME
        )


@dataclass
class Ring (JewelryObject):
    OBJECT_NAME: str = "Pierścionek"

    X_AXIS_MOVEMENT_PER_FRAME: int = 25
    Y_AXIS_MOVEMENT_PER_FRAME: int = 10

    def __post_init__(self):
        self.X_AXIS_MOVEMENT_PER_FRAME = 25
        self.Y_AXIS_MOVEMENT_PER_FRAME = 10


@dataclass
class Necklace (JewelryObject):
    OBJECT_NAME: str = "Naszyjnik"

    X_AXIS_MOVEMENT_PER_FRAME: int = 15
    Y_AXIS_MOVEMENT_PER_FRAME: int = 10


@dataclass
class Bracelet (JewelryObject):
    OBJECT_NAME: str = "Bransoletka"

    X_AXIS_MOVEMENT_PER_FRAME: int = 15
    Y_AXIS_MOVEMENT_PER_FRAME: int = 10


@dataclass
class ObjectTracker:

    TrackingWindow: tuple[int, int] = field(init=True, default=(1, 1))

    necklaces: list[Necklace] = field(init=False, default_factory=list)
    rings: list[Ring] = field(init=False, default_factory=list)
    bracelets: list[Bracelet] = field(init=False, default_factory=list)

    detectedObjects: dict[str, int] = field(init=False, default_factory=dict)

    def trackObjects(
            self,
            rings_key_points: tuple[cv2.KeyPoint] = tuple(),
            necklaces_key_points: tuple[cv2.KeyPoint] = tuple(),
            bracelets_key_points: tuple[cv2.KeyPoint] = tuple()
    ):
        self.__track_rings(rings_key_points)

    def __track_rings(self, key_points: tuple[cv2.KeyPoint] = tuple()):
        distanceTable = ObjectTracker.__get_distance_table(
            self.rings,
            key_points
        )

        for KP_id in distanceTable:
            if not distanceTable[KP_id]:
                self.__add_new_ring(key_points[KP_id])
                continue

            ring_id_closest_match, x_dist, y_dist = ObjectTracker.__get_closest_position_match(
                distanceTable[KP_id]
            )

            if (
                x_dist <= Ring.X_AXIS_MOVEMENT_PER_FRAME and
                y_dist <= Ring.Y_AXIS_MOVEMENT_PER_FRAME
            ):
                self.__append_ring_position(
                    ring_id_closest_match,
                    key_points[KP_id]
                )
            else:
                self.__add_new_ring(key_points[KP_id])

        # print(len(self.rings))
        return None

    def __add_new_ring(self, key_point: tuple[float, float]):
        self.rings.append(Ring())
        self.__append_ring_position(-1, key_point)
        return None

    def __append_ring_position(self, ring_id: int, key_point: tuple[float, float]):
        (
            self
            .rings[ring_id]
            .appendPositions(key_point)
        )
        return None

    @staticmethod
    def __get_distance_table(
        objectToTrack: list[Ring | Necklace | Bracelet],
        key_points: tuple[cv2.KeyPoint] = tuple()
    ) -> dict[int, tuple[float, float]]:

        distanceTable = {}
        for KP_id in range(0, len(key_points)):
            tempDistanceTable = {}

            for object_id in range(0, len(objectToTrack)):
                distance = (
                    objectToTrack[object_id]
                    .calculateDistance(key_points[KP_id])
                )
                tempDistanceTable[object_id] = distance

            distanceTable[KP_id] = tempDistanceTable

        return distanceTable

    @staticmethod
    def __get_closest_position_match(distances: dict[int, tuple[float, float]]):
        ring_id: int = min(
            distances,
            key=lambda k: distances[k]
        )

        x_dist, y_dist = distances[ring_id]

        return (ring_id, x_dist, y_dist)
