import numpy as np
import copy
class Point():
    """
    This is a class which hold contour points and the relevant information
    we require

    """
    def __init__(self, pixel_coords=None, sop_instance_uid=None, weight = 1):
        if pixel_coords == None:
            self.pixel =np.empty(2)
        else:
            self.pixel = pixel_coords

        self.sop_instance_uid = sop_instance_uid
        self.coordinates = np.empty(3)
        self.weight = weight




    def __eq__(self, other):

        if self.pixel == other.pixel:
            if self.sop_instance_uid ==other.sop_instance_uid:
                 equal = True
            else:
                equal =False
        else:equal =False
        return equal


    def deep_copy_point(self):
        new_point = Point()
        new_point.pixel = copy.deepcopy(self.pixel)
        new_point.sop_instance_uid = copy.deepcopy(self.sop_instance_uid)
        new_point.coordinates = copy.deepcopy(self.coordinates)
        new_point.weight = copy.deepcopy(self.weight)
        return  new_point


class Frame():
    def __init__(self, image_id,position, orientation, pixel_spacing,
                 image = None, subpixel_resolution = 1):
        self.position = position
        self.orientation = orientation
        self.pixel_spacing = pixel_spacing
        self.subpixel_resolution = subpixel_resolution
        self.image = image

        self.time_frame = 1
        self.slice = None
        self.image_id = image_id

    def get_affine_matrix(self, scaling = False):
        spacing = self.pixel_spacing
        image_position_patient = self.position
        image_orientation_patient = self.orientation
        # Translation
        T = np.identity(4)
        T[0:3, 3] = image_position_patient
        # Rotation
        R = np.identity(4)
        R[0:3, 0] = image_orientation_patient[0:3]
        R[0:3, 1] = image_orientation_patient[3:7]
        R[0:3, 2] = np.cross(R[0:3, 0], R[0:3, 1])
        T = np.dot(T, R)
        # scale
        if scaling:
            S = np.identity(4)
            S[0, 0] = spacing[1]
            S[1, 1] = spacing[0]
            T = np.dot(T, S)
        return T