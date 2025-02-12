import warnings
import os
import numpy as np
from scipy.spatial import ckdtree

from bivme.fitting.Slice import Slice, Point


class Contours:
    def __init__(self, dict_of_points=None, dict_of_frame=None, log=False):
        if dict_of_points == None:
            self.points = {}
        else:
            self.points = dict_of_points  # disctionary of collections of points,
        # indexed by contour name, a contour will contains a list of points
        if dict_of_frame == None:
            self.slice = {}
        else:
            self.slice = dict_of_frame  # dictionary of frames indexed by frame uid

        self.nb_frames = 0
        self.nb_points = 0
        self._time_uid_map = {}
        self._space_uid_map = {}
        self.log = log

    def read_gp_files(self, gp_file, metadata, time_frame=None):
        """
        add  by A. Mira 02/2020
        read contours from a gp file
        If no time frame is specified, all existent time frames are read
        Args:
            gp_file: file with contours as exported from CVI42Extraction
            metadata: file with image information as exported from CVI42Extraction
            time_frame: ist of time frames to read

        Returns:
            contour object
        """

        if not os.path.exists(gp_file):
            warnings.warn("\033[2;37;45m Contour files does not exist")
            return

        try:
            # read GPfile
            first_line = np.genfromtxt(gp_file, max_rows=1, usecols=(0, 1, 2))
            skip_header = 1 if np.isnan(first_line).any() else 0
            data_set = np.genfromtxt(
                gp_file, skip_header=skip_header, usecols=(0, 1, 2), dtype=float
            )
            contour_type = np.genfromtxt(
                gp_file, skip_header=skip_header, usecols=(3), dtype=str
            )

            gp_slice_id = np.genfromtxt(
                gp_file, skip_header=skip_header, usecols=(4), dtype=int
            )
            weight = np.genfromtxt(
                gp_file, skip_header=skip_header, usecols=(5), dtype=float
            )
            file_time_frames = np.genfromtxt(
                gp_file, skip_header=skip_header, usecols=(6), dtype=float
            )

        except ValueError:
            print("\033[2;37;45m Wrong file format: {0}\n".format(gp_file))

        valid_contour_index = np.zeros_like(file_time_frames).astype(bool)
        if not (time_frame is None):
            if np.isscalar(time_frame):
                time_frame = [time_frame]
            for time in time_frame:
                valid_contour_index[file_time_frames == time] = True
        else:
            valid_contour_index = []
        # select just the time frames we are interested in
        if np.sum(valid_contour_index) > 0:
            data_set = data_set[valid_contour_index, :]
            contour_type = contour_type[valid_contour_index]
            file_time_frames = file_time_frames[valid_contour_index]
            gp_slice_id = gp_slice_id[valid_contour_index]

        # read metadata file
        used_slice_id, index = np.unique(gp_slice_id, return_index=True)
        corresponding_time = file_time_frames[index]
        slices_uid = np.genfromtxt(metadata, usecols=(0), dtype=str)
        slices_id = np.genfromtxt(metadata, usecols=(2), dtype=int)

        try:
            position = np.genfromtxt(metadata, usecols=(6, 7, 8), dtype=float)
            orientation = np.genfromtxt(
                metadata, usecols=(10, 11, 12, 13, 14, 15), dtype=float
            )
            pixel_spacing = np.genfromtxt(metadata, usecols=(17, 18), dtype=float)

        except:
            position = np.genfromtxt(metadata, usecols=(4, 5, 6), dtype=float)
            orientation = np.genfromtxt(
                metadata, usecols=(8, 9, 10, 11, 12, 13), dtype=float
            )
            pixel_spacing = np.genfromtxt(metadata, usecols=(15, 16), dtype=float)

        for index, point in enumerate(data_set):
            new_point = Point()
            new_point.coordinates = point
            new_point.weight = weight[index]
            new_point.sop_instance_uid = slices_uid[slices_id == gp_slice_id[index]][0]
            new_point.time_frame = file_time_frames[index]
            self.add_point(contour_type[index], new_point)

        # increment contours points which don't need sampling
        for index, slice_id in enumerate(slices_id):
            if slice_id in used_slice_id:
                time_frame = corresponding_time[slice_id == used_slice_id][0]
                new_slice = Slice(
                    slice_id, position[index], orientation[index], pixel_spacing[index]
                )
                new_slice.time_frame = time_frame
                self.add_slice(slices_uid[index], new_slice)

    def merge_contours(self, new_contour):
        for new_c in new_contour.list_contour_types():
            if new_c in self.list_contour_types():
                self.points[new_c] = self.points[new_c] + new_contour.points[new_c]
            else:
                self.points.update({new_c: new_contour.points[new_c]})
            self.nb_points = self.nb_points + len(new_contour.points[new_c])

        for new_slice in new_contour.list_slice_uids():
            if not (new_slice in self.list_slice_uids()):
                self.slice.update({new_slice: new_contour.slice[new_slice]})
                self.nb_slices = self.nb_slices + 1
        self._compute_slices()
        self._compute_time_frames()
        return self

    def get_slice_points(self, contour, slice_uids=None):
        """
        returns the points associated with a list of frame uids amd a
        contour type. If slice_uids is not defined, it returns the contour
        for all existing frames
        Args:
            contour: string, type of contour
            slice_uids:

        Returns:
            list_of_index: list of ints corresponding to the points index,
                            as found in self.points[contour]
            list_of_points: list of points objects corresponding to the
                        frames with sop_instance_uid ='slice_uids' and
                        contour_type ='contour'

        """
        list_of_points = []
        list_of_index = []

        if contour not in self.list_contour_types():
            return list_of_index, list_of_points

        if slice_uids == None:
            slice_uids = []
        if not isinstance(slice_uids, list):
            slice_uids = [slice_uids]
        if len(slice_uids) == 0:
            slice_uids = self.list_slice_uids()

        for point_index, point in enumerate(self.points[contour]):
            if point.sop_instance_uid in slice_uids:
                list_of_index.append(point_index)
                list_of_points.append(point)

        return list_of_index, list_of_points

    def update_points(self, contour, list_of_points, list_of_index):
        """
        Update points object of a given contour
        Args:
            contour: string, type of contour to be updated
            list_of_points: list of points objects
            list_of_index: list of ints, indx of points to be updated in
                            self.points[contour] list
        """
        if contour not in self.list_contour_types():
            return
        for index, point_index in enumerate(list_of_index):
            try:
                self.points[contour][point_index] = list_of_points[index]
            except:
                warnings.warn("\033[2;37;45m Invalid points index")

    def get_pixels(self, contour):
        if contour in self.list_contour_types():
            return [x.pixel for x in self.points[contour]]
        else:
            []

    def update_pixels(self, contour, pixels, lst_of_index):
        # updates pixel val for a list of points
        if contour not in self.list_contour_types():
            return
        for index, point_index in enumerate(lst_of_index):
            try:
                self.points[contour][point_index].pixel = pixels[index]
            except:
                ValueError("Invalid list of index")

    def get_point_coordinates(self, points=[]):
        if not isinstance(points, list):
            points = [points]
        if len(points) == 0:
            contours = self.list_contour_types()
            for contour in contours:
                points = points + self.points[contour]

        coordinates_list = [x.coordinates for x in points]
        return np.reshape(coordinates_list, (len(coordinates_list), 3))

    def get_coordinates_by_index(self, contour, index=[]):
        # get the points coordicates for a contour type indexing by point index
        if contour not in self.list_contour_types():
            return np.empty(0)
        if len(index) == 0:
            index = list(range(len(self.points[contour])))
        if not isinstance(index, list):
            index = [index]

        return np.array([self.points[contour][x].coordinates for x in index])

    def update_points_coordinates(self, contour, lst_coords, lst_index):
        if contour not in self.list_contour_types():
            return

        for index, point_index in enumerate(lst_index):
            try:
                self.points[contour][point_index].coordinates = lst_coords[index]
            except:
                ValueError("Invalid list of indexes")

    def get_slice_pixels(self, contour, slice_uid):
        index_list = []
        pixel_list = []
        if contour not in self.list_contour_types():
            return index_list, pixel_list

        for point_index, point in enumerate(self.points[contour]):
            if point.sop_instance_uid == slice_uid:
                index_list.append(point_index)
                pixel_list.append(point.pixel)

        return index_list, np.reshape(pixel_list, (len(pixel_list), 2))

    def get_slice_points_coordinates(self, contour, slice_uids):
        """
        input:
            slice_uid:  the frame unique ID number
        output:
            index_list: a list on ints giving the index of
                        points corresponding to the frame
            coordinate_list: nx3 array of giving the spatial
                            coordinates of points.
        """
        index_list = []
        coordinates_list = []

        if contour not in self.list_contour_types():
            return index_list, coordinates_list

        index_list, frame_points = self.get_slice_points(contour, slice_uids)
        coordinates_list = [x.coordinates for x in frame_points]

        return index_list, np.reshape(coordinates_list, (len(coordinates_list), 3))

    def list_sop_instance_uid(self, contours=[]):
        if not isinstance(contours, list):
            contours = [contours]
        if len(contours) == 0:
            contours = self.list_contour_types()

        existing_contours = self.list_contour_types()
        existing_uids = []
        for contour in contours:
            if contour in existing_contours:
                existing_uids = existing_uids + [
                    point.sop_instance_uid for point in self.points[contour]
                ]
            else:
                print(" Contour {0} is empty \n".format(contour))

        return np.unique(existing_uids)

    def add_slice(self, slice_uid, new_slice):
        self.slice.update({slice_uid: new_slice})
        self.nb_slices += 1
        # set int id using the trigger time # will be used to select
        # points at different time frames
        self._compute_slices()
        self._compute_time_frames()

    def add_point(self, contour, new_point):
        if contour in self.list_contour_types():
            self.points[contour].append(new_point)
        else:
            self.points.update({contour: [new_point]})
        self.nb_points += 1

    def delete_point(self, contour, index_to_delete):
        if contour in self.list_contour_types():
            new_list = [
                self.points[contour][i]
                for i in range(len(self.points[contour]))
                if i not in index_to_delete
            ]
            self.points[contour] = new_list

            self.nb_points = self.nb_points - len(index_to_delete)

    def set_contour_weights(self, contour, weights):
        contour_points = self.points[contour]
        if contour in self.list_contour_types():
            if isinstance(weights, float):
                for point in contour_points:
                    point.weight = weights
            elif isinstance(weights, list):
                if len(weights) == len(contour_points):
                    for index, point in enumerate(contour_points):
                        point.weight = weights[index]

    def list_slice_uids(self):
        return list(self.slice.keys())

    def get_slice_position(self, slice_uid):
        if slice_uid in self.slice.keys():
            return self.slice[slice_uid].position
        else:
            return None

    def get_subpixel_resolution(self, slice_uid):
        if slice_uid in self.slice.keys():
            return self.slice[slice_uid].subpixel_resolution
        else:
            return None

    def get_slice_orientation(self, slice_uid):
        if slice_uid in self.slice.keys():
            return self.slice[slice_uid].orientation
        else:
            return None

    def get_slice_pixel_spacing(self, slice_uid):
        if slice_uid in self.slice.keys():
            return self.slice[slice_uid].pixel_spacing
        else:
            return None

    def list_contour_types(self):
        return list(self.points.keys())

    def list_slice_uids_at_timeframe(self, timeframe):
        if np.isscalar(timeframe):
            timeframe = [timeframe]
        if not isinstance(timeframe, list):
            timeframe = list(timeframe)

        slice_uids = []
        for frame in timeframe:
            if frame in self._time_uid_map.keys():
                slice_uids = slice_uids + list(self._time_uid_map[frame])

        return slice_uids

    def list_slice_uids_at_slice(self, spaceframe):
        if spaceframe in self._space_uid_map:
            return list(self._space_uid_map[spaceframe])
        else:
            return []

    def get_timeframe_points(self, contour, time_frame):
        if np.isscalar(time_frame):
            time_frame = [time_frame]
        if not isinstance(time_frame, list):
            time_frame = list(time_frame)

        point_lst = []
        index_lst = []
        for time in time_frame:
            if contour not in self.list_contour_types():
                return index_lst, point_lst

            for slice_uid in self.list_slice_uids_at_timeframe(time):
                local_index_lst, local_point_lst = self.get_slice_points(
                    contour, slice_uid
                )
                point_lst = point_lst + local_point_lst
                index_lst = index_lst + local_index_lst

        return index_lst, point_lst

    def get_slice_points(self, contour, spaceframe):
        point_lst = []
        index_lst = []
        if contour not in self.list_contour_types():
            return

        for slice_uid in self.list_slice_uids_at_timeframe(spaceframe):
            local_index_lst, local_point_lst = self.get_slice_points(contour, slice_uid)
            point_lst = point_lst + local_point_lst
            index_lst = index_lst + local_index_lst

        return index_lst, point_lst

    def get_timeframe_points_coordinates(self, contour, time_frame):
        index_lst, timeframe_points = self.get_timeframe_points(contour, time_frame)
        coordinates_lst = [x.coordinates for x in timeframe_points]
        return index_lst, np.reshape(coordinates_lst, (len(coordinates_lst), 3))

    def slice_points_coordinates(self, contour, slice):
        index_lst, timeframe_points = self.get_slice_points(contour, slice)
        coordinates_lst = [x.coordinates for x in timeframe_points]
        return index_lst, np.reshape(coordinates_lst, (len(coordinates_lst), 3))

    def list_time_frames(self):
        return list(self._time_uid_map.keys())

    def list_slices(self):
        return list(self._space_uid_map.keys())

    def get_contour_point_coordinates(self, contour_type):
        """
        Return the set of points corresponding to a contour type
        from all time frames and space frames
        input:
            contour_type:  string describing the type of contour
        output:
                coordinates_lst: nx3  array with points spatial
                            coordinates
        """

        if contour_type in self.points.keys():
            return [x.coordinates for x in self.points[contour_type]]
        else:
            return []

    def _list_unique_tirigger_time(self):
        """Returns a list of parameter lists that determine the
        spatial position of the image frame.
        Each position is uniquely determined by
        frame_orientation (6 parameters) and
        frame_position (3 parameters).
        Each entry in the output list is a concatenation of
        frame_orientation and frame_position into 9
        parameters."""
        position_pairs = []
        for slice_uid in self.list_slice_uids():
            position_pairs.append(self.slice[slice_uid].trigger_time)

        return sorted(np.unique(position_pairs))

    def _compute_time_frames(self):
        """arange time frames according to the trigger time
        associate each frame with an integer.
        Also computes a dictionary maping the time frame with a
        list of frames uids"""

        # sort frames by frame time

        self._time_uid_map = {}

        # group time frames using the trigger time
        for slice_uid in self.list_slice_uids():
            frame_number = self.slice[slice_uid].time_frame
            if frame_number in self._time_uid_map.keys():
                self._time_uid_map[int(frame_number)].append(slice_uid)
            else:
                self._time_uid_map.update({int(frame_number): [slice_uid]})

    def _list_unique_slice_positions(self):
        """Returns a list of lists of scalars that determine the
        spatial position of the image frame. Each position is uniquely
        determined by frame_orientation (6 parameters) and
        frame_position (3 parameters).
        Each entry in the output list is a concatenation of
        frame_orientation and frame_position into 9 parameters."""
        position_pairs = []
        for slice_uid in self.slice.keys():
            position_pairs.append(
                list(self.slice[slice_uid].position)
                + list(self.slice[slice_uid].orientation)
            )
        # sorted by z pozition
        return sorted(np.unique(position_pairs, axis=0), key=lambda x: x[2])

    def _compute_slices(self):
        """Assign an integer frame number to each frame position to
        make them easier to refer to.  frame numbers correspond to the
        indices of the output of `self.list_unique_slices()`."""
        self._space_uid_map = {}
        unique_positions = self._list_unique_slice_positions()
        for slice_uid in self.list_slice_uids():
            frame_position = np.concatenate(
                (self.slice[slice_uid].position, self.slice[slice_uid].orientation)
            )
            slice = np.where((unique_positions == frame_position).all(axis=1))[0][0]
            self.slice[slice_uid].slice = slice
            if slice not in self._space_uid_map.keys():
                self._space_uid_map.update({slice: [slice_uid]})
            else:
                self._space_uid_map[slice].append(slice_uid)

    def _list_trigger_time(self):
        trigger_time = []
        for slice_uid in self.list_slice_uids():
            trigger_time.append(self.slice[slice_uid].trigger_time)

        return list(trigger_time)

    def _list_instance_number(self):
        image_ids = []
        for slice_uid in self.list_slice_uids():
            image_ids.append(self.slice[slice_uid].instance_number)

        return list(image_ids)

    def find_timeframe_septum(self, time_frames=None, tol=None):
        """
        Computes RV_inserts for a given time frame.
        the RV insert points are added to the list of
        corresponding frame points
        If time frame is not defined the septum will be computed for all
        existent time frames
        Args:
            time_frames: scalar or list of time frames to be processed
            tol: tolerance for the distance between the RV edo and LV epi

        Returns:

        """
        if time_frames is None:
            time_frames = self.list_time_frames()
        else:
            if not isinstance(time_frames, list):
                time_frames = [time_frames]

        for time_frame in time_frames:
            valid_time_frames = time_frame
            if not isinstance(valid_time_frames, list):
                valid_time_frames = [valid_time_frames]
            if len(valid_time_frames) == 0:
                valid_time_frames = self.list_time_frames()
            else:
                valid_time_frames = valid_time_frames
            contours_to_use = [
                "SAX_RV_EPICARDIAL",
                "SAX_RV_ENDOCARDIAL",
                "LAX_LV_EPICARDIAL",
                "LAX_RV_ENDOCARDIAL",
            ]

            # check for the interesting contours exists
            existing_contours = self.list_contour_types()
            skip = [contour not in existing_contours for contour in contours_to_use[:2]]
            if not (np.any(skip)):
                skip = [
                    contour not in existing_contours for contour in contours_to_use[-2:]
                ]
                if np.any(skip):
                    print(
                        "\033[2;37;45m The epicardial or/and rvendocardial contours "
                        "have not been found\n"
                    )
                    return

            index_to_delete = []
            if self.log:
                print("Computing Septum and Free Wall points")
            for slice in self.list_slices():
                # find the frames uids corresponding to the given
                # space frame and time frame
                valid_slices = [
                    x
                    for x in self.list_slice_uids_at_slice(slice)
                    if x in self.list_slice_uids_at_timeframe(time_frame)
                ]
                for frame in valid_slices:
                    if self.log:
                        print("     Frame: {0}".format(frame))
                    self.find_slice_septum(frame, tol=tol)

    def clean_LAX_contour(self, time_frame=None):
        """
        # Clean the points from the endocardial contours
        # between the two valve points
        # If no time_frame difined, the points will be deleted for all
        # existent time frames.
        Args:
            time_frame: scalar or list of time frames to be processed
        """
        if time_frame is None:
            time_frame = self.list_time_frames()
        if not isinstance(time_frame, list):
            time_frame = [time_frame]

        valve_contours = ["LAX_RV_EXTENT", "LAX_RV_EXTENT", "LAX_LV_EXTENT"]
        lax_contours = ["LAX_RV_FREEWALL", "LAX_RV_ENDOCARDIAL", "LAX_LV_ENDOCARDIAL"]

        for time in time_frame:
            frames = self.list_slice_uids_at_timeframe(time)
            for index, contour in enumerate(lax_contours):
                delete_index = []
                for frame in frames:
                    pc_index, contour_points = self.get_slice_points(contour, frame)
                    _, extent_points = self.get_slice_points(
                        valve_contours[index], frame
                    )

                    if len(extent_points) == 3:
                        valve_dist = np.linalg.norm(
                            extent_points[1].coordinates - extent_points[0].coordinates
                        )
                        for p_index, point in enumerate(contour_points):
                            distance1 = np.linalg.norm(
                                extent_points[1].coordinates - point.coordinates
                            )
                            distance2 = np.linalg.norm(
                                extent_points[0].coordinates - point.coordinates
                            )
                            point_dist = distance1 + distance2
                            if abs(point_dist - valve_dist) < 3:
                                delete_index.append(pc_index[p_index])

                self.delete_point(contour, delete_index)

    def clean_LAX_3ch_contour(self, time_frame=None):
        """
        # Clean the points from the endocardial contours
        # between the two valve points
        # If no time_frame difined, the points will be deleted for all
        # existent time frames.
        Args:
            time_frame: scalar or list of time frames to be processed


        """
        if time_frame is None:
            time_frame = self.list_time_frames()
        if not isinstance(time_frame, list):
            time_frame = [time_frame]
        valve_contours = ["AORTA_VALVE"]

        lax_contours = ["LAX_RV_EPICARDIAL"]

        for time in time_frame:
            frames = self.list_slice_uids_at_timeframe(time)
            for index, contour in enumerate(lax_contours):
                delete_index = []
                for frame in frames:
                    pc_index, contour_points = self.get_slice_points(contour, frame)

                    _, extent_points = self.get_slice_points(
                        valve_contours[index], frame
                    )

                    if len(extent_points) == 2:
                        for p_index, point in enumerate(contour_points):
                            delete_index.append(pc_index[p_index])

                self.delete_point(contour, delete_index)

                # self.delete_point(contour,delete_index)

    def find_slice_septum(self, slice_uid, tol=None):
        """This replaces find inserts. It takes the RV contour and works
        out whats septal and whats free wall.
        From this we can find the inserts by taking the angle between the
        septal points and the LV_ENDOCARDIAL centroid. The largest angle is the
        RV inserts
        Author: A. MIRA 01/2020 adapted from
                Kathleen Gilbert
                Date : 24/05/2017"""
        # find points corresponding to RV and LV_ENDOCARDIAL on to the
        # given frame.
        lv_contours = ["SAX_LV_EPICARDIAL", "LAX_LV_EPICARDIAL"]
        rv_contours = ["SAX_RV_ENDOCARDIAL", "LAX_RV_ENDOCARDIAL"]
        septum_contours = ["SAX_RV_SEPTUM", "LAX_RV_SEPTUM"]
        rvfw_contours = ["SAX_RV_FREEWALL", "LAX_RV_FREEWALL"]

        if tol is None:
            tol = min(self.get_slice_pixel_spacing(slice_uid))

        index_to_delete = []
        for idx in range(len(lv_contours)):
            lv_points_index, lv_points = self.get_slice_points(
                lv_contours[idx], slice_uid
            )

            rv_points_index, rv_points = self.get_slice_points(
                rv_contours[idx], slice_uid
            )

            if len(lv_points_index) == 0:
                # if there are no points corresponding to left or right
                # ventricle returns empty lists
                if len(rv_points_index) != 0:
                    for point_index, point in enumerate(rv_points):
                        new_point = point.deep_copy_point()
                        if rv_contours[idx] == "SAX_RV_ENDOCARDIAL":
                            self.add_point("SAX_RV_OUTLET", new_point)
                        else:
                            self.add_point("LAX_RV_ENDOCARDIAL", new_point)
                continue
            elif len(rv_points_index) == 0:
                continue

            # compute the distance between points from rv edocardial and lv
            # epicadial
            lv_points_tree = ckdtree.cKDTree(self.get_point_coordinates(lv_points))
            distance, _ = lv_points_tree.query(self.get_point_coordinates(rv_points))
            # save the index where the distance is smaller than pixel
            # spacing. These points will be used to create new points
            # corresponding to rv septum and free wall
            (septum_index,) = np.where(distance < tol)
            for point_index, point in enumerate(rv_points):
                new_point = point.deep_copy_point()
                if point_index in septum_index:
                    self.add_point(septum_contours[idx], new_point)
                else:
                    self.add_point(rvfw_contours[idx], new_point)
            # the point should be deleted at the end of the loop to keep
            # the same points index
            # too slow to delet points
            # all points from lv coreresponding to septum will be deleted
            rv_points_tree = ckdtree.cKDTree(self.get_point_coordinates(rv_points))
            inverse_distance, _ = rv_points_tree.query(
                self.get_point_coordinates(lv_points)
            )
            (lv_index_to_delete,) = np.where(inverse_distance < tol)
            index_to_delete = [lv_points_index[i] for i in lv_index_to_delete]
            # rv_index_to_delete = [rv_points_index[i] for i in
            #                        septum_index]

            # index_to_delete = [y for x in index_to_delete for y in x]
            self.delete_point(lv_contours[idx], index_to_delete)

        # mlab.figure()
        # mlab.points3d([self.points[ind].coordinates[ 0] for
        #                ind in septum_index],
        #               [self.points[ind].coordinates[ 1] for
        #                ind in septum_index],
        #               [self.points[ind].coordinates[ 2] for
        #                ind in septum_index],
        #               opacity=0.5,
        #               scale_factor=0.8,
        #               scale_mode='vector', mode='sphere', color=(1,1,1))
        # mlab.show()

    def find_timeframe_septum_inserts(self, time_frame=[]):
        valid_time_frames = time_frame
        if not isinstance(valid_time_frames, list):
            valid_time_frames = [valid_time_frames]
        if len(valid_time_frames) == 0:
            valid_time_frames = self.list_time_frames()
        else:
            valid_time_frames = valid_time_frames

        contours_to_use = ["SAX_RV_FREEWALL", "LAX_RV_FREEWALL"]
        epi_contour = ["SAX_LV_EPICARDIAL", "LAX_LV_EPICARDIAL"]
        existing_contours = self.list_contour_types()

        skip = [contour not in existing_contours for contour in contours_to_use]
        if np.all(skip):
            print(
                "\033[2;37;45m RV Free Wall contour points must "
                "be computed for time frame {0} \n To compute RV_SEPTUM and RV_FREEWALL "
                "use contour.find_timeframe_septum()",
                format(time_frame),
            )
            return

        existing_index = np.where([not x for x in skip])[0]
        skip = [epi_contour[i] not in existing_contours for i in existing_index]
        if np.all(skip):
            print(
                "\033[2;37;45m The RV epicardial contour is not defined "
                "for time frame "
                ".\n Insert points could not be computed"
            )
            return
        if self.log:
            print("Computing Septum Inserts points")
        for space_slice in self.list_slices():
            # find the frames uids corresponding to the given
            # space frame and time frame
            for time_frame in valid_time_frames:
                valid_slices = [
                    x
                    for x in self._space_uid_map[space_slice]
                    if x in self._time_uid_map[time_frame]
                ]
                for slice in valid_slices:
                    if self.log:
                        print("     Frame: {0}".format(slice))
                    self.find_slice_septum_insert(slice)

    def find_slice_septum_insert(self, slice_uid):
        # rv_fw_contour = ['SAX_RV_FREEWALL','LAX_RV_FREEWALL']
        # epi_contour = ['SAX_RV_EPICARDIAL','LAX_LV_EPICARDIAL']

        _, septum_points = self.get_slice_points("SAX_RV_SEPTUM", slice_uid)
        if len(septum_points) == 0:
            _, septum_points = self.get_slice_points("LAX_RV_SEPTUM", slice_uid)
            _, free_points = self.get_slice_points("LAX_RV_FREEWALL", slice_uid)
            if len(septum_points) == 0:
                return

        insert_1, insert_2 = self.find_extreme_points(septum_points)
        new_point1 = insert_1.deep_copy_point()
        self.add_point("RV_INSERT", new_point1)
        new_point2 = insert_2.deep_copy_point()
        self.add_point("RV_INSERT", new_point2)
        # from mayavi import mlab
        # mlab.figure()
        # mlab.points3d([x.coordinates[ 0] for
        #               x in septum_points],
        #              [x.coordinates[ 1] for
        #               x in septum_points],
        #              [x.coordinates[ 2] for
        #               x in septum_points],
        #              opacity=0.5,
        #              scale_factor=0.8,
        #              scale_mode='vector', mode='sphere', color=(1,1,1))
        # mlab.points3d(new_point1.coordinates[0] ,
        #              new_point1.coordinates[1] ,
        #              new_point1.coordinates[2] ,
        #              opacity=0.5,
        #              scale_factor=0.8,
        #              scale_mode='vector', mode='sphere', color=(1, 0, 1))
        # mlab.points3d(new_point2.coordinates[0],
        #              new_point2.coordinates[1],
        #              new_point2.coordinates[2],
        #              opacity=0.5,
        #              scale_factor=0.8,
        #              scale_mode='vector', mode='sphere', color=(1, 0, 1))
        # mlab.show()

    @staticmethod
    def find_extreme_points(points_list):
        # compute the distance between two sets of points

        # the closest points will be found on the two edges of rv
        # edocardial wall. We should find the closes point on both
        # edges. Therefore the intra points distance is computed to
        # separate the closest points into two sets.

        farthest_points_index = [
            np.argmax(
                [
                    np.linalg.norm(point1.coordinates - point2.coordinates)
                    for point2 in points_list
                ]
            )
            for point1 in points_list
        ]
        farthest_points = [points_list[i] for i in farthest_points_index]
        max_distance = [
            np.linalg.norm(
                points_list[ind].coordinates - farthest_points[ind].coordinates
            )
            for ind in range(len(points_list))
        ]

        extreme_index = np.argmax(max_distance)

        # clasify points in points belowing to the same edge and
        # oposite edge

        point_one = points_list[extreme_index]
        point_two = points_list[farthest_points_index[extreme_index]]
        return point_one, point_two

    def compute_3D_coordinates(self, contours=[], timeframe=[]):
        if not isinstance(contours, list):
            contours = [contours]
        if len(contours) == 0:
            contours = self.list_contour_types()
        if self.log:
            print("Compute contour points coordinates")
        points = []
        existing_contours = self.list_contour_types()
        for contour in contours:
            if contour in existing_contours:
                points = points + list(
                    self.convert_2D_to_3D_timeframe_contour(contour, timeframe)
                )
        return points

    def convert_2D_to_3D_timeframe_contour(self, contour, timeframe=[]):
        if len(self.slice) == 0:
            ValueError("Metadata is missing. Load dicom metadata first")

        valid_time_frames = timeframe
        if not isinstance(valid_time_frames, list):
            valid_time_frames = [valid_time_frames]
        if len(valid_time_frames) == 0:
            valid_time_frames = self.list_time_frames()

        if self.log:
            print("     Contour: {0}".format(contour))
        changed_points_index = []
        for frame in self.list_slice_uids_at_timeframe(valid_time_frames):
            pixels_indx, pixels = self.get_slice_pixels(contour, frame)
            if len(pixels_indx) == 0:
                continue
            if self.log:
                print("         Frame {0}".format(frame))

            image_position = self.get_slice_position(frame)
            image_orientation = self.get_slice_orientation(frame)
            pixel_spacing = self.get_slice_pixel_spacing(frame)
            subpixel_res = self.get_subpixel_resolution(frame)
            points_coordinates = self.from_2d_to_3d(
                pixels, image_orientation, image_position, pixel_spacing, subpixel_res
            )

            self.update_points_coordinates(contour, points_coordinates, pixels_indx)

            changed_points_index = changed_points_index + pixels_indx

        return self.get_coordinates_by_index(contour, changed_points_index)

    @staticmethod
    def from_2d_to_3d(
        p2, ImageOrientation, ImagePosition, PixelSpacing, subpixel_resolution
    ):
        """# Convert indices of a pixel in a 2D image in space to 3D coordinates.
        #	Inputs
        #		ImageOrientation
        #		ImagePosition
        #		PixelSpacing
        #		subpixel_resolution
        #	Outputs
        #		P3:  3D points
        """
        # if points2D.
        points2D = np.array(p2) / subpixel_resolution

        S = np.eye(4)
        S[0, 0] = PixelSpacing[1]
        S[1, 1] = PixelSpacing[0]
        S = np.matrix(S)

        R = np.identity(4)
        R[0:3, 0] = ImageOrientation[
            0:3
        ]  # col direction, i.e. increases with row index i
        R[0:3, 1] = ImageOrientation[
            3:7
        ]  # row direction, i.e. increases with col index j
        R[0:3, 2] = np.cross(R[0:3, 0], R[0:3, 1])

        T = np.identity(4)
        T[0:3, 3] = ImagePosition

        F = np.identity(4)
        F[0:1, 3] = -0.5

        T = np.dot(T, R)
        T = np.dot(T, S)
        Transformation = np.dot(T, F)

        pts = np.ones((len(points2D), 4))
        pts[:, 0:2] = points2D
        pts[:, 2] = [0] * len(points2D)
        pts[:, 3] = [1] * len(points2D)

        Px = np.dot(Transformation, pts.T)
        p3 = Px[0:3, :] / (np.vstack((Px[3, :], np.vstack((Px[3, :], Px[3, :])))))
        p3 = p3.T

        x = np.asarray(p3[:, 0])
        y = np.asarray(p3[:, 1])
        z = np.asarray(p3[:, 2])

        P3 = np.zeros((len(points2D), 3))
        P3[:, 0] = x[:, 0]
        P3[:, 1] = y[:, 0]
        P3[:, 2] = z[:, 0]

        return P3

    def find_timeframe_valve_landmarks(self, time_frame=[]):
        """Grabs the 2 corners marking the valve openings in atrial contours.
         The 3 points below are the last 3 points read from the XML file for this contour.
         2 of the points mark the corners of the valve opening and the other
         1 is a point between the corners (not necessarily a straight line between the 2 corners).

        Adapted from : OldCode
                    Author: Kathleen Gilbert
                     Date : 20170525"""
        if self.log:
            print("Searching landmarks for mitral and tricuspid valve")
        if len(time_frame) == 0:
            time_frame = self.list_time_frames()
        if np.isscalar(time_frame):
            time_frame = [time_frame]

        existent_contour = self.list_contour_types()
        for phase in time_frame:
            contours_to_use = ["LAX_LV_EXTENT", "LAX_RV_EXTENT"]
            valid_contour = [
                contour not in existent_contour for contour in contours_to_use
            ]
            if np.all(valid_contour):
                contours_to_use = ["LAX_LA", "LAX_RA"]

                valid_contour = [
                    contour not in existent_contour for contour in contours_to_use
                ]

            if np.all(valid_contour):
                warnings.warn(
                    "\033[2;37;45m {0} and {1} contours are "
                    "missing".format(contours_to_use[0], contours_to_use[1])
                )

            # LA atrial slices
            # points should already be ordered (file read-in order)

            for space_slice in self.list_slices():
                valid_slices = [
                    x
                    for x in self._space_uid_map[space_slice]
                    if x in self._time_uid_map[phase]
                ]
                for frame in valid_slices:
                    if self.log:
                        print("     Frame: {0}".format(frame))
                    # check if there are some artria LAX points to be sure that the valves points
                    # are selected in a 4CH. 2CH or 3CH view
                    la_points = self.get_slice_points(contours_to_use[0], frame)
                    ra_points = self.get_slice_points(contours_to_use[1], frame)
                    if len(la_points[0]) > 0 or len(ra_points[0]) > 0:
                        self.find_slice_valve_landmarks(frame, contours_to_use)

    def find_slice_valve_landmarks(self, slice_uid, contours_to_use):
        # select just the rv epicadium contour
        for contour_type in contours_to_use:
            _, mitral_points = self.get_slice_points(contour_type, slice_uid)

            if len(mitral_points) > 3:
                mitral_points = mitral_points[-3:]

                points_3d = self.get_point_coordinates(mitral_points)

                intra_distance = [
                    np.linalg.norm(point1 - point2)
                    for point1 in points_3d
                    for point2 in points_3d
                ]
                points_index = divmod(np.argmax(intra_distance), 3)

            elif len(mitral_points) == 3:
                points_index = range(2)
            else:
                continue

            for index in points_index:
                new_point = mitral_points[index].deep_copy_point()
                if contour_type == contours_to_use[0]:
                    #          _, la_points = self.get_slice_points(
                    #              "LAX_LV_ENDOCARDIAL", slice_uid)
                    #          if len(la_points)>0:
                    self.add_point("MITRAL_VALVE", new_point)
                if contour_type == contours_to_use[1]:
                    # _, la_rv_points = self.get_slice_points(
                    # "LAX_RV_ENDOCARDIAL", slice_uid)
                    # check if the same frame have also LA RV points
                    # to avoid la ra contours without points on tricuspid
                    # valve
                    # if len(la_rv_points)>0 :
                    self.add_point("TRICUSPID_VALVE", new_point)

    def find_apex_landmark(self, time_frame=[]):
        """ "This function takes the laxLVextnedPoints and creates an
        LV_ENDOCARDIAL apex point from the long axis slice

        Adapted from : OldCode
                    Author: Kathleen Gilbert
                     Date : 20170525"""

        if np.isscalar(time_frame):
            time_frame = [time_frame]
        if len(time_frame) == 0:
            time_frame = self.list_time_frames()
        if self.log:
            print("Searching for apex ...")

        contours_to_use = ["LAX_LV_EXTENT", "LAX_LA", "LAX_RA"]
        existent_contour = self.list_contour_types()
        invalid_contour = [
            contour not in existent_contour for contour in contours_to_use
        ]

        if np.all(invalid_contour):
            warnings.warn(
                "\033[2;37;45m {0} and {1} contours is "
                "missing".format(contours_to_use[0], contours_to_use[1])
            )

            return
        if invalid_contour[0]:
            warnings.warn(
                " \033[2;37;45m {0} contours is missing".format(contours_to_use[0])
            )
            return

        if invalid_contour[1]:
            warnings.warn(
                "\033[2;37;45m {0} contours is missing".format(contours_to_use[1])
            )
            return
        for frame in time_frame:
            # find frame with laxLvExtentPoints
            extent_idx, extent_points = self.get_timeframe_points(
                contours_to_use[0], frame
            )

            # Select unique frames
            space_slices_uid = [x.sop_instance_uid for x in extent_points]
            space_slices_uid = np.unique(space_slices_uid)
            # LA atrial slices
            # points should already be ordered (file read-in order)
            all_extent_points = []
            all_lala_coords = []
            # we are searching for the centroid of LALA contour
            # the apex will be the farthest LVExtent point from the LALA
            # contour.
            for slice_uid in space_slices_uid:
                lala_index, lala_coordinates = self.get_slice_points_coordinates(
                    contours_to_use[1], slice_uid
                )
                lara_index, lara_coordinates = self.get_slice_points_coordinates(
                    contours_to_use[2], slice_uid
                )

                if len(lala_coordinates) > 0:
                    _, extent_points = self.get_slice_points(
                        contours_to_use[0], slice_uid
                    )

                    all_extent_points = all_extent_points + list(extent_points)
                    all_lala_coords = all_lala_coords + list(lala_coordinates)

            if len(all_extent_points) > 0:
                centroid = np.mean(all_lala_coords, axis=0)
                distance = [
                    np.linalg.norm(x.coordinates - centroid) for x in all_extent_points
                ]
                apex_index = np.argmax(distance)
                new_point = all_extent_points[apex_index].deep_copy_point()
                new_point.time_frame = frame
                self.add_point("APEX_POINT", new_point)

    def define_apex_landmark(self, time_frame=None):
        """
        When LV extent is not defined ht eapez is computed as
        (slice_min - slice_int) as the apex slice position,
        and use the center of LV contour on slice_min as the in-plate position
        """
        if np.isscalar(time_frame):
            time_frame = [time_frame]
        if time_frame is None:
            time_frame = self.list_time_frames()

        for phase in time_frame:
            slices_uids = self.list_slice_uids_at_timeframe(phase)
            positions = np.array([self.get_slice_position(x) for x in slices_uids])
            sorted_position_index = np.argsort(positions[:, 2])
            frame1_uid = slices_uids[sorted_position_index[0]]
            frame2_uid = slices_uids[sorted_position_index[1]]
            _, frame1_points = self.get_slice_points_coordinates(
                "SAX_LV_ENDOCARDIAL", frame1_uid
            )
            _, frame2_points = self.get_slice_points_coordinates(
                "SAX_LV_ENDOCARDIAL", frame2_uid
            )
            # some frames dont have points for the LV_ENDO contour
            i = 0
            while frame1_points.shape[0] == 0:
                i = i + 1
                frame1_uid = slices_uids[sorted_position_index[i]]
                frame2_uid = slices_uids[sorted_position_index[i + 1]]
                _, frame1_points = self.get_slice_points_coordinates(
                    "SAX_LV_ENDOCARDIAL", frame1_uid
                )

            centroid1 = np.mean(frame1_points, axis=0)

            frame1_position = self.get_slice_position(frame1_uid)
            frame2_position = self.get_slice_position(frame2_uid)
            new_position = 2 * frame1_position - frame2_position
            new_point = Point()
            new_slice = copy.deepcopy(self.slice[frame1_uid])

            new_slice.image_id = len(self.slice) + 1
            new_slice.position = new_position

            new_point.coordinates = centroid1 - (frame2_position - frame1_position)
            new_point.sop_instance_uid = "apex_virtual_frame_{0}".format(phase)

            self.add_point("APEX_POINT", new_point)
            self.add_slice("apex_virtual_frame_{0}".format(phase), new_slice)

    def find_pulmonary_valve_landmarks(self, timeframe):
        time_frame = 0  # laxLVextnedPoints only drawn at ED
        if self.log:
            print("Searching for Pulmonary valve ...")

        contour_to_use = "SAX_RV_OUTLET"
        existent_contour = self.list_contour_types()

        if contour_to_use not in existent_contour:
            print("\033[2;37;45m {0} contour is missing \n".format(contour_to_use))
            print("\033[2;37;45m Pulmonary valve could not be found \n")
            return

        slices_uid = self.list_slice_uids_at_timeframe(time_frame)
        contour_slices = self.list_sop_instance_uid(contour_to_use)

        valid_slices = [x for x in contour_slices if x in slices_uid]
        if len(valid_slices) == 0:
            print("\033[2;37;45m  Pulmonary valve could not be found \n")
            return
        max_distance = 10000

        for frame in valid_slices:
            points_index, frame_points = self.get_slice_points(contour_to_use, frame)
            distance = np.array(
                [
                    np.linalg.norm(x.coordinates - y.coordinates)
                    for x in frame_points
                    for y in frame_points
                ]
            )
            if max(distance) < max_distance:  # maximal distance is the
                # diameter of the valve, upper the frame is smaller is the
                # diameter. Here we are selecting the upper frame of points
                pulmonary_slice_uid = frame
                max_distance = max(distance)

        if max_distance > 30:
            return

        indx, pulmonary_points = self.get_slice_points(
            contour_to_use, pulmonary_slice_uid
        )
        for point in pulmonary_points:
            new_point = point.deep_copy_point()
            self.add_point("PULMONARY_VALVE", new_point)

        self.delete_point(contour_to_use, indx)
