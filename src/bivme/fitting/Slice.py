import numpy as np
import copy


class Point:
    """
    This is a class which hold contour points and the relevant information
    we require

    """

    def __init__(
        self, pixel_coords: np.array=None, sop_instance_uid: str=None, weight: float=1, time_frame: int=None
    ) -> None:
        if pixel_coords == None:
            self.pixel = np.empty(2)
        else:
            self.pixel = pixel_coords

        self.sop_instance_uid = sop_instance_uid
        self.coordinates = np.empty(3)
        self.weight = weight
        self.time_frame = time_frame

    def __eq__(self, other) -> bool:
        if np.all(self.pixel == other.pixel):
            if self.sop_instance_uid == other.sop_instance_uid:
                equal = True
            else:
                equal = False
        else:
            equal = False
        return equal

    def deep_copy_point(self):
        new_point = Point()
        new_point.pixel = copy.deepcopy(self.pixel)
        new_point.sop_instance_uid = copy.deepcopy(self.sop_instance_uid)
        new_point.coordinates = copy.deepcopy(self.coordinates)
        new_point.weight = copy.deepcopy(self.weight)
        new_point.time_frame = copy.deepcopy(self.time_frame)
        return new_point

class Slice:
    def __init__(
        self,
        image_id: int,
        position: np.ndarray,
        orientation: np.ndarray,
        pixel_spacing: np.ndarray,
        image: np.ndarray=None,
        subpixel_resolution: int=1,
    ) -> None:
        self.position = position
        self.orientation = orientation
        self.pixel_spacing = pixel_spacing
        self.subpixel_resolution = subpixel_resolution
        self.image = image

        self.time_frame = 1
        self.slice = None
        self.image_id = image_id

    def get_affine_matrix(self, scaling: bool=False) -> np.ndarray:
        spacing = self.pixel_spacing
        image_position_patient = self.position
        image_orientation_patient = self.orientation
        # Translation
        translation = np.identity(4)
        translation[0:3, 3] = image_position_patient
        # Rotation
        rotation = np.identity(4)
        rotation[0:3, 0] = image_orientation_patient[0:3]
        rotation[0:3, 1] = image_orientation_patient[3:7]
        rotation[0:3, 2] = np.cross(rotation[0:3, 0], rotation[0:3, 1])
        translation = np.dot(translation, rotation)
        # scale
        if scaling:
            scale = np.identity(4)
            scale[0, 0] = spacing[1]
            scale[1, 1] = spacing[0]
            translation = np.dot(translation, scale)
        return translation
