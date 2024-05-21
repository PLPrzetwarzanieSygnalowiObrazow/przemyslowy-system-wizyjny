from dataclasses import dataclass, field
import cv2
import numpy
from dependencies.objectsDefinition import Ring, Necklace, Earings
from dependencies.draw import Draw
X = 0
Y = 1


@dataclass
class ObjectTracker:

    COLOR_ASSIGNMENT_TO_OBJECT_TYPES = {
        Necklace: Draw.COLOR_BLUE,
        Ring: Draw.COLOR_RED,
        Earings: Draw.COLOR_GREEN
    }
    necklaces: list[Necklace] = field(init=False, default_factory=list)
    rings: list[Ring] = field(init=False, default_factory=list)
    earings: list[Earings] = field(init=False, default_factory=list)

    __analyzed_frames: int = field(init=False, default=0)

    def trackObjects(
            self,
            rings_key_points: tuple[cv2.KeyPoint] = tuple(),
            necklaces_key_points: tuple[cv2.KeyPoint] = tuple(),
            earings_key_points: tuple[cv2.KeyPoint] = tuple(),
            frame_to_draw: numpy.ndarray = None
    ) -> numpy.ndarray:
        '''
            Public method to track objects in the video frame.
            Invokes tracking methods for know object types:
                - rings
                - necklaces
                - earings

            Returns a frame with selected objects.
        '''
        self.__count_analyzed_frames()

        # track rings
        frame_to_draw = self.__track_objects_of_given_type(
            Ring,
            self.rings,
            rings_key_points,
            frame_to_draw
        )

        # track necklaces
        frame_to_draw = self.__track_objects_of_given_type(
            Necklace,
            self.necklaces,
            necklaces_key_points,
            frame_to_draw
        )

        # track earings
        frame_to_draw = self.__track_objects_of_given_type(
            Earings,
            self.earings,
            earings_key_points,
            frame_to_draw
        )

        return frame_to_draw

    def printTrackingReport(self) -> None:
        self.__clean_up_phantom_objects()
        print("Analyzed frames: ", self.__analyzed_frames)
        print("Found rings: ", len(self.rings))
        print("Found necklaces: ", len(self.necklaces))
        print("Found earings: ", len(self.earings))

    def __clean_up_phantom_objects(self) -> None:
        '''
            Method to remove wrongly identified objects found due to any artifacts on the frame.
        '''

        objectTypesToCleanup: dict[Ring | Necklace | Earings, list[Ring | Necklace | Earings]] = {
            Ring: self.rings,
            Necklace: self.necklaces,
            Earings: self.earings
        }

        for object_type in objectTypesToCleanup:
            for object in objectTypesToCleanup[object_type]:
                if object.getFoundOnFrames() < object_type.MARK_AS_INVISIBLE_AFTER_MISSING_ON_FRAMES:
                    objectTypesToCleanup[object_type].remove(object)

    def __count_analyzed_frames(self):
        self.__analyzed_frames = self.__analyzed_frames + 1

    @ staticmethod
    def __track_objects_of_given_type(
        object: Ring | Necklace | Earings,
        objectsToTrack: list[Ring | Necklace | Earings],
        key_points: tuple[cv2.KeyPoint] = tuple(),
        frame_to_draw: numpy.ndarray = None
    ):
        '''
            Method to perform necessary operations to track rings
        '''

        # mark objects on the frame
        frame_to_return = Draw.keyPoints(
            frame=frame_to_draw,
            keyPoints=key_points,
            color=ObjectTracker.COLOR_ASSIGNMENT_TO_OBJECT_TYPES[object]
        )

        # if object to track are earings group them into pairs before further processing
        if object is Ring:
            key_points = Earings.groupEaringsIntoPairs(key_points)

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
                    object,
                    objectsToTrack,
                    key_points[KP_id]
                )
                continue

            # get closest object to current key point
            closest_ring_id, x_dist, y_dist = ObjectTracker.__get_closest_position_match(
                distanceTable[KP_id]
            )

            # check if found closest object meets criteria to be identified as this the same
            if ObjectTracker.__object_assignment_validation(
                object,
                objectsToTrack,
                closest_ring_id,
                x_dist, y_dist
            ):

                # if yes append to positions
                ObjectTracker.__append_object_position(
                    objectsToTrack,
                    closest_ring_id,
                    key_points[KP_id]
                )
            else:
                # if not add new entry
                ObjectTracker.__add_new_object(
                    object,
                    objectsToTrack,
                    key_points[KP_id]
                )

        # increment counter for each object which has not been found
        ObjectTracker.__increment_missing_on_frames(objectsToTrack)

        return frame_to_return

    @ staticmethod
    def __object_assignment_validation(
        object: Ring | Necklace | Earings,
        object_list: list[Ring | Necklace | Earings],
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
                object.X_AXIS_MOVEMENT_ERROR *
                thresholdMux
            )
        )

        # check if distance on Y axis is less or equal to acceptable error
        # related to conveyor belt movement, multiplied by number of frames,
        # where particular object was not found
        Y_axis_condition: bool = (
            y_distance <= (
                object.Y_AXIS_MOVEMENT_ERROR *
                thresholdMux
            )
        )

        # returns true if both conditions are met
        return (X_axis_condition and Y_axis_condition)

    @ staticmethod
    def __add_new_object(
        object: Ring | Necklace | Earings,
        listToAppend: list[Ring | Necklace | Earings],
        key_point: tuple[float, float]
    ):

        # create new object in list
        listToAppend.append(object())

        # append positions for new object
        ObjectTracker.__append_object_position(listToAppend, -1, key_point)

        return None

    @ staticmethod
    def __append_object_position(
        listToAppend: list[Ring | Necklace | Earings],
        object_id: int,
        key_point: tuple[float, float]
    ):
        # append positions of object with given id
        (
            listToAppend[object_id]
            .appendPositions(key_point)
        )

        return None

    @ staticmethod
    def __get_distance_table(
        objectsToTrack: list[Ring | Necklace | Earings],
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
            for object_id in range(0, len(objectsToTrack)):

                # consider only visible objects
                if objectsToTrack[object_id].isVisible():

                    # get distance between current object coordinates and key point in format
                    # (x_dist, y_dist)
                    distance = (
                        objectsToTrack[object_id]
                        .calculateDistance(key_points[KP_id])
                    )

                    # append dict for current key point
                    distanceTableCurrentKeyPoint[object_id] = distance

            # append distances for last calculated key point
            distanceTable[KP_id] = distanceTableCurrentKeyPoint

        return distanceTable

    @ staticmethod
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

    @ staticmethod
    def __reset_append_flag(objectList:  list[Ring | Necklace | Earings]) -> None:

        for object in objectList:
            object.resetAppendFlag()

        return None

    @ staticmethod
    def __increment_missing_on_frames(objectList:  list[Ring | Necklace | Earings]) -> None:

        for object in objectList:
            object.incrementMissingOnFrames()

        return None
