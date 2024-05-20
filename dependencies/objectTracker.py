from dataclasses import dataclass, field
import cv2

X = 0
Y = 1


@dataclass
class JewelryObject:

    positions: list[cv2.KeyPoint] = field(init=False, default_factory=list)
    visible: bool = field(init=False, default=True)
    X_AXIS_MOVEMENT_ERROR: int = field(init=False, default=10)
    Y_AXIS_MOVEMENT_ERROR: int = field(init=False, default=10)

    X_AXIS_MOVEMENT_PER_FRAME: int = field(init=False, default=None)
    Y_AXIS_MOVEMENT_PER_FRAME: int = field(init=False, default=None)
    __missing_on_frames: int = field(init=False, default=0)
    __found_on_frames: int = field(init=False, default=1)
    __appended: bool = field(init=False, default=True)

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
        if not self.__appended:
            self.__missing_on_frames = self.__missing_on_frames + 1

    def calculateDistance(self, key_point: cv2.KeyPoint = None):

        lastPosition = self.positions[-1]
        movement_estimation = self.getAcceptableMovement()
        x_distance = abs(
            key_point.pt[X] - (lastPosition.pt[X] + movement_estimation[X])
        )
        y_distance = abs(
            key_point.pt[Y] - (lastPosition.pt[Y] + movement_estimation[Y])
        )
        return (x_distance, y_distance)

    def getAcceptableMovement(self) -> tuple[int, int]:
        return (
            self.__missing_on_frames * self.X_AXIS_MOVEMENT_PER_FRAME,
            self.__missing_on_frames * self.Y_AXIS_MOVEMENT_PER_FRAME
        )


@dataclass
class Ring (JewelryObject):
    OBJECT_NAME: str = "Pierścionek"

    # X_AXIS_MOVEMENT_PER_FRAME: int = 4
    # Y_AXIS_MOVEMENT_PER_FRAME: int = 1

    def __post_init__(self):
        self.X_AXIS_MOVEMENT_PER_FRAME = 3
        self.Y_AXIS_MOVEMENT_PER_FRAME = 1


@dataclass
class Necklace (JewelryObject):
    OBJECT_NAME: str = "Naszyjnik"

    X_AXIS_MOVEMENT_PER_FRAME: int = 10
    Y_AXIS_MOVEMENT_PER_FRAME: int = 2


@dataclass
class Bracelet (JewelryObject):
    OBJECT_NAME: str = "Bransoletka"

    X_AXIS_MOVEMENT_PER_FRAME: int = 4
    Y_AXIS_MOVEMENT_PER_FRAME: int = 1


@dataclass
class ObjectTracker:

    necklaces: list[Necklace] = field(init=False, default_factory=list)
    rings: list[Ring] = field(init=False, default_factory=list)
    bracelets: list[Bracelet] = field(init=False, default_factory=list)

    __analyzed_frames: int = field(init=False, default=0)

    def __count_analyzed_frames(self):
        self.__analyzed_frames = self.__analyzed_frames + 1

    def trackObjects(
            self,
            rings_key_points: tuple[cv2.KeyPoint] = tuple(),
            necklaces_key_points: tuple[cv2.KeyPoint] = tuple(),
            bracelets_key_points: tuple[cv2.KeyPoint] = tuple()
    ):
        self.__count_analyzed_frames()
        self.__track_objects_of_given_type(self.rings, rings_key_points)

    def clean_up_phantom_objects(self):
        '''
            Method to remove wrongly identified objects found due to any artifacts on the frame.
        '''
        print("Analyzed frames: ", self.__analyzed_frames)
        for ring in self.rings:
            if ring.getFoundOnFrames() < 10:
                print("Found on frames: ", ring.getFoundOnFrames())
                self.rings.remove(ring)

    def __track_necklaces(self):
        pass

    @staticmethod
    def __track_objects_of_given_type(
        objectsToTrack: Ring | Necklace | Bracelet,
        key_points: tuple[cv2.KeyPoint] = tuple()
    ):
        '''
            Method to perform necessary operations to track rings
        '''
        # get distance tables
        distanceTable = ObjectTracker.__get_distance_table(
            objectsToTrack,
            key_points
        )

        # reset append flag on each object in the list,
        # to be able to increment missing on frame counter after checking all key points
        ObjectTracker.__reset_append_flag(objectsToTrack)

        # loop through key points in distance table
        for KP_id in distanceTable:

            # if dict for particular key point is empty it means that object list is empty,
            # so create new object entry and continue to next key point
            if not distanceTable[KP_id]:
                ObjectTracker.__add_new_object(
                    Ring(),
                    objectsToTrack,
                    key_points[KP_id]
                )
                continue

            # get closest object to current key point
            closest_ring_id, x_dist, y_dist = ObjectTracker.__get_closest_position_match(
                distanceTable[KP_id]
            )

            # check if found closest object meets criteria to be identified as this the same
            if ObjectTracker.__object_assignment_validation(objectsToTrack, closest_ring_id, x_dist, y_dist):

                # if yes append to positions
                ObjectTracker.__append_object_position(
                    objectsToTrack,
                    closest_ring_id,
                    key_points[KP_id]
                )
            else:
                # if not add new entry
                print("ID: ", closest_ring_id)
                print("X dist: ", x_dist)
                print("Y dist: ", y_dist)
                ObjectTracker.__add_new_object(
                    Ring(),
                    objectsToTrack,
                    key_points[KP_id]
                )

        # increment counter for each object which has not been found
        ObjectTracker.__increment_missing_on_frames(objectsToTrack)

        return None

    @staticmethod
    def __object_assignment_validation(
        object_list: list[JewelryObject],
        objectID: int,
        x_distance: float,
        y_distance: float
    ) -> bool:
        '''
            Method to validate if object situated the closest to particular key point is meeting
            the criteria for assigning it to that object
        '''
        # get number of frames where object was missing
        # +1 in case of object which has number of missing frames 0,
        # to avoid multiplication by 0
        thresholdMux = object_list[objectID].getMissingOnFrames() + 1

        # check if distance on X axis is less or equal to acceptable error
        # related to conveyor belt movement, multiplied by number of frames,
        # where particular object was not found
        X_axis_condition: bool = (
            x_distance <= (
                Ring.X_AXIS_MOVEMENT_ERROR *
                thresholdMux
            )
        )

        # check if distance on Y axis is less or equal to acceptable error
        # related to conveyor belt movement, multiplied by number of frames,
        # where particular object was not found
        Y_axis_condition: bool = (
            y_distance <= (
                Ring.Y_AXIS_MOVEMENT_ERROR *
                thresholdMux
            )
        )

        # returns true if both conditions are met
        return (X_axis_condition and Y_axis_condition)

    @staticmethod
    def __add_new_object(
        object: Ring | Necklace | Bracelet,
        listToAppend: list[Ring | Necklace | Bracelet],
        key_point: tuple[float, float]
    ):

        # create new object in list
        listToAppend.append(object)

        # append positions for new object
        ObjectTracker.__append_object_position(listToAppend, -1, key_point)

        return None

    @staticmethod
    def __append_object_position(
        listToAppend: list[Ring | Necklace | Bracelet],
        object_id: int,
        key_point: tuple[float, float]
    ):
        # append positions of object with given id
        (
            listToAppend[object_id]
            .appendPositions(key_point)
        )
        return None

    @staticmethod
    def __get_distance_table(
        objectToTrack: list[Ring | Necklace | Bracelet],
        key_points: tuple[cv2.KeyPoint] = tuple()
    ) -> dict[int, tuple[float, float]]:
        '''
            Calculates distance dict between objects and key points, peer-to-peer
            returns dict in format:
            {
                <ring_object_id>: {
                    <key_point_id>: (<distance_X_axis>, <distance_Y_axis>)
                }
            }
        '''
        distanceTable = {}

        # loop through provided key_points
        for KP_id in range(0, len(key_points)):
            distanceTableCurrentKeyPoint = {}

            # loop through object list
            for object_id in range(0, len(objectToTrack)):

                # get distance between current object coordinates and key point in format
                # (x_dist, y_dist)
                distance = (
                    objectToTrack[object_id]
                    .calculateDistance(key_points[KP_id])
                )

                # append dict for current key point
                distanceTableCurrentKeyPoint[object_id] = distance

            # append distances for last calculated key point
            distanceTable[KP_id] = distanceTableCurrentKeyPoint

        return distanceTable

    @staticmethod
    def __get_closest_position_match(distances: dict[int, tuple[float, float]]):
        '''
            Method to find closest distance in provided distance dict
            returns tuple: ring_ID, distance on X axis, distance on Y axis
        '''
        # get ring class object id, where sum of x distance and y distance is the lowest
        ring_id: int = min(
            distances,
            key=lambda k: (abs(distances[k][0]) + abs(distances[k][1]))
        )

        # read out those distances
        x_dist, y_dist = distances[ring_id]

        return (ring_id, x_dist, y_dist)

    @staticmethod
    def __reset_append_flag(objectList:  list[Ring | Necklace | Bracelet]) -> None:
        for object in objectList:
            object.resetAppendFlag()

        return None

    @staticmethod
    def __increment_missing_on_frames(objectList:  list[Ring | Necklace | Bracelet]) -> None:
        for object in objectList:
            object.incrementMissingOnFrames()

        return None
