import xml.etree.ElementTree as ET

import pydicom as dcm
import Contours

import numpy as np
import os
import re
from copy import deepcopy
import xml.etree.ElementTree as ET
from parse_cvi42_xml import *
from xml.dom import minidom
import sys
sys.path.append('C:/Users/ldt18/Desktop/Dev_HPC/BiV_Modelling_v2')  # path to where the BiVFitting folder is located
from BiVFitting import Frame,Point

contour_name_map = { 'larvendocardialContour': 'LAX_RV_ENDOCARDIAL',
    				'larvepicardialContour': 'LAX_RV_EPICARDIAL ',
   					 'laendocardialContour': 'LAX_LV_ENDOCARDIAL',
   					'laepicardialContour':  'LAX_LV_EPICARDIAL' ,

    				"sarvendocardialContour": 'SAX_RV_ENDOCARDIAL',
    				 "sarvepicardialContour": 'SAX_RV_EPICARDIAL',
    				"saendocardialContour":'SAX_LV_ENDOCARDIAL',
    				"saepicardialContour": 'SAX_LV_EPICARDIAL',
					 'laxLaExtentPoints' : 'LAX_LA_EXTENT',
					 'laxRaExtentPoints' : 'LAX_RA_EXTENT',
					 'laxRvExtentPoints': 'LAX_RV_EXTENT',
					 'laxLvExtentPoints' : 'LAX_LV_EXTENT',
					 'laraContour': 'LAX_RA',
					 'lalaContour':'LAX_LA',
					 'saepicardialOpenContour':'SAX_LV_EPICARDIAL',
					 'saendocardialOpenContour': 'SAX_LV_ENDOCARDIAL',
					 'AorticValveDiameter':'AORTA_VALVE',
					 'PulmonaryValveDiameter':'PULMONARY_VALVE',
					 'AV':'AORTA_VALVE',
					 'MV':'MITRAL_VALVE'
					 }

# Charlène Mauger - 2019
class CVI42XML():
	# This class reads an xml file from CVI42 and extracts the 2D points 
	def __init__(self, filepath_contours =None, dcm_path = None,
				 dicom_extension = '.dcm',
				 convert_3D = True, log = False ):
		'''Input:
		#		filepath_xml: path to xml file
		#		contour_file: path to output file
		# Output:
		#		File containing the 2D pixel coordinates. The file is structured as follows:
		#		point coordidnates x and y, label, SOPInstanceUID '''
		
		if filepath_contours is None:
			return
		parser = ET.XMLParser(encoding="utf-8")
		tree = ET.parse(filepath_contours,parser= parser)
		self.root = tree.getroot()
		# contour_points, dcm_uid, contour_type, frame_number, time_frame
		# should be the same length and are corresponding by index
		self.contour = Contours.Contours(log = log)
		previous_item = []

		# CVI42 extracts all sorts of contours.

		# Extract contours
		errorCode = self.search_contours(filepath_contours, previous_item)
		if os.path.isdir(dcm_path):
			self.search_metadata(dcm_path, dicom_extension= dicom_extension)
		elif os.path.isfile(dcm_path):
			self.seart_txt_metadata(dcm_path)
		else:
			print('Wrong path')
			raise ValueError
		if convert_3D == True:
			self.contour.compute_3D_coordinates()
		# out.close()
	def set_contour(self, contour):
		self.contour = contour


	def search_dicom_files(self,dcm_path, dicom_extension = '.dcm'):
		# search for dicom files asssociated with contours points
		file_uids = {}
		file_uid_frame = {}
		
		dcm_uids = self.contour.list_sop_instance_uid()
		for dir_name, subdir_list, file_list in os.walk(dcm_path):
			total_files_uid = []
			instance_number = []
			position = []
			series_number = []
			read = 0

			for filename in file_list:

				if dicom_extension in filename:  # check whether the file's DICOM
					file_dcm = os.path.join(dir_name, filename)
					dicom_data = dcm.read_file(file_dcm, force=True)

					try:
						file_uid = dicom_data.SOPInstanceUID
						position.append(np.asfarray(
						dicom_data.ImagePositionPatient))
						instance =int(dicom_data.InstanceNumber)
						series =float(dicom_data.SeriesNumber)
					except:
						continue

					instance_number.append(instance)
					series_number.append(series)
					read =read+1
					total_files_uid.append(file_uid)

					if (file_uid in dcm_uids) and (file_uid not in
													   file_uids):

						file_uids.update({file_uid:file_dcm})
			unique_snb = np.unique(series_number)
			series_number = np.array(series_number)
			# in case that different series with different lengths are stored in
			# the same folder, the images stack needs to be sorted by series first

			if len(unique_snb != len(series_number)):
				# classify by series number

				for serie in unique_snb:
					serie_index = np.where(series_number == serie)[0]
					# the frames need to be computed using the full stack of SAx or LAX
					# images, for each folder the frame need to be computed
					# using the trigger time and instance number we recover the
					# slice images in the aquisition order

					serie_uids = np.array([total_files_uid[x] for x in serie_index])
					serie_instance_nb =  np.array([instance_number[x] for x in serie_index])
					serie_position = np.array([position[x] for x in serie_index])
					sorted_index = np.argsort(serie_position,axis = 0)
					if len(sorted_index) ==0:
						continue


					slice_len = int(np.sum(
						np.sum(np.equal(serie_position,serie_position[0]),axis =1)==3))
					for index, uid in enumerate(serie_uids):
						if uid in file_uids.keys():
							frame = int(serie_instance_nb[index]%slice_len)

							if frame ==0:
								frame = slice_len
							file_uid_frame.update({uid:frame-1})


		existing_uid = [(x in file_uids.keys()) for x in dcm_uids]
		# Make sure that contour files is a subset of dicom files from
		# directory, so that there is a dicom corresponding to each contour point
		if not all(existing_uid):  # if
			raise ValueError(' dicom file/UID are missing'
						'Insufficient dicom data. Only a subset of points '
						  'can be extracted')

		return  file_uids, file_uid_frame


	def search_metadata(self ,dcm_path,
						dicom_extension = '.dcm'):

		# create an empty list
		dicom_files, files_frames = self.search_dicom_files(dcm_path,
													   dicom_extension)



		for dcm_uid in dicom_files.keys():

			dicom_data = dcm.read_file(dicom_files[dcm_uid])

			image_position = np.array([float(x)
							  for x in
							  dicom_data.ImagePositionPatient])
			image_orientation =np.array([float(x) for x in
							dicom_data.ImageOrientationPatient])
			pixel_spacing = np.array([float(x) for x in
							 dicom_data.PixelSpacing])
			#  collect metadata associate with the frame

			image_nb = self.contour.nb_frames+1
			new_frame = Frame(image_nb,image_position, image_orientation,
							pixel_spacing, None,self.subpixel_resolution)
			new_frame.time_frame = files_frames[dcm_uid]
			self.contour.add_frame(dcm_uid, new_frame)

		if not(len(self.contour.list_frame_uids()) ==
			   len(self.contour.list_sop_instance_uid())):
			raise ValueError('Missing images from dicom stack')
			
	def seart_txt_metadata(self ,file_name):
		
		#  it using csv read
		if not os.path.exists(file_name):
			return
		lines = {}

		with open(file_name, 'rt') as in_file:
			for line in in_file:
				info = re.split('\s+', line)
				lines.update({info[0]:info[1:]})

		try:
			info =lines[list(lines.keys())[0]]
			index_imPos = np.where(['ImagePositionPatient' in x
									for x in info])[0][0] + 1
			index_imOr = \
			np.where(['ImageOrientationPatient' in x
					  for x in info])[0][0] + 1
			timeFrame_index = int(np.where(['timeFrame' in x
									   for x in info])[0][0] + 1)
			index_pixel_spacing = (np.where(['PixelSpacing' in x
											for x in info])[0][0] + 1)
		except:
			raise ValueError('Wrong file format in reading dicom metadata')

		image_nb = 0

		for dcm_uid in self.contour.list_sop_instance_uid():
			if not(dcm_uid in lines.keys()):
				print(dcm_uid)
				raise ValueError('Missing images from dicom stack')

			image_position = np.array(
				[float(x) for x in lines[dcm_uid][index_imPos:index_imPos+3]])
			image_orientation =np.array([float(x) for x in
										 lines[dcm_uid][index_imOr:index_imOr+6]])
			pixel_spacing = np.array([float(x) for x in
									  lines[dcm_uid][index_pixel_spacing:index_pixel_spacing+2]])
			timeFrame = float(lines[dcm_uid][timeFrame_index])
			##  collect metadata associate with the frame

			new_frame = Frame(image_nb,image_position, image_orientation,
							pixel_spacing, None,self.subpixel_resolution)
			image_nb = image_nb+1
			new_frame.time_frame = timeFrame
			self.contour.add_frame(dcm_uid, new_frame)




	def search_contours(self, root, parent_item):
		"""This function searches the contours and calls itself to keep searching the children
		#Input: 
		#	root: root element
		#	previous_item: parent of the child
		#	out: output file
		#	list_of_excluded_contours: contour types we are excluding
		#Output:
		#	errorCode: error code. 0 if nbo issues. 2 otherwise
		#"""
		errorCode = 0

		# if previous_item != "ManualSeries":
		self.get_points(root,parent_item)

		'''
			if children.getchildren()!=[]:
				errorCode = self.search_contours(children.getchildren(),
												 CurrentItem)

				if errorCode == 2:
					return errorCode 

			# previous_item = CurrentItem
		'''

		return 2


	def get_points(self, root, parent):
		""" This function extracts the 2D coordinates
		#Input:
		#	root: root element
		#	parent: parent of the child
		#	out: output file
		#	list_of_excluded_contours: contour types we are excluding
		#Output:
		#	None
		"""
		
		dom = minidom.parse(root)
		uid_contours = {}
		traverseNode(dom, uid_contours)

		points_lst = []

		for uid, contours in uid_contours.items():
			for key in contours: 
				self.subpixel_resolution = 1

				if key in contour_name_map.keys():
					for element in contours[key]:
						new_point = Point([int(element[0]),int(element[1])], uid)
						self.contour.add_point(
									contour_name_map[key], new_point)

						#points_lst.append((element[0], element[1]))	


		'''
	
		for children in root:
			if children.getchildren()!=[]:
				point_type = children.get(
					'{http://www.circlecvi.com/cvi42/Workspace/Hash/}key')
				label = None
				for grandchildren in children:
					CurrentItem = grandchildren.get(
						'{http://www.circlecvi.com/cvi42/Workspace/Hash/}key')
					if CurrentItem == "SubpixelResolution":
						self.subpixel_resolution = float(grandchildren.text)

					if CurrentItem == 'Label':
						label = grandchildren.text

					"add by A. Mira 01/2020"
					# selec the contours types we are interested in
					# For our purpose, we do not need the papillary muscles and
					# ROI information so we exclude them
					valid_point_type  =not(('Musc' in point_type) or \
										('Ref' in point_type) or \
									   ('flow' in point_type) )


					if CurrentItem =="Points" and valid_point_type:

						pointLevel = grandchildren.getchildren()
						for points in pointLevel:
							individual_point = points.getchildren()
							new_point = Point([
											int(individual_point[1].text),
											int(individual_point[0].text)],
								parent)
				
							if 'Roi' in point_type:
								self.contour.add_point(
									'MANUAL_CONTOUR', new_point)




							elif point_type in contour_name_map.keys():
								self.contour.add_point(
									contour_name_map[point_type], new_point)

				if (label in contour_name_map.keys()) and (
						'MANUAL_CONTOUR' in
						self.contour.points.keys()):
					self.contour.add_point(contour_name_map[label],
										   deepcopy(self.contour.points[
														'MANUAL_CONTOUR'][
														0]))
					self.contour.add_point(contour_name_map[label],
										   deepcopy(self.contour.points[
														'MANUAL_CONTOUR'][
														-1]))
					del self.contour.points['MANUAL_CONTOUR']
				if 'MANUAL_CONTOUR' in self.contour.points.keys():

					del self.contour.points['MANUAL_CONTOUR']

		'''

		return points_lst

	def save_contours_uid_file(self, file_name):
		if not os.path.exists(os.path.dirname(file_name)):
			os.makedirs(os.path.dirname(file_name))

		out = open(file_name, 'w')

		for contour_type in self.contour.list_contour_types():
			for c_point in self.contour.points[contour_type]:
				out.write(str(c_point.pixel[0])
					  + " "
					  + str(c_point.pixel[1])
					  + " " + c_point.contour_type + " " +
						  c_point.sop_instance_uid + '\n')
		out.close()

	def export_contour_points(self, file_name, time_frames= []):


		if not isinstance(time_frames, list):
			time_frames = [time_frames]
		if len(time_frames) == 0:
			time_frames = self.contour.list_time_frames()
		else:
			time_frames = time_frames

		if not os.path.exists(os.path.dirname(file_name)):
			os.makedirs(os.path.dirname(file_name))
		out = open(file_name, 'w')
		out.write('{0:}\t{1}\t{2}\t'.format('x', 'y', 'z')
				  + '{0}\t'.format('contour type')
				  + '{0}\t{1}\t{2}'.format('sliceID','weight','time frame')
				  +'\n')

		for frame_uid in self.contour.list_frame_uids():
			if self.contour.frame[frame_uid].time_frame in time_frames:
				for contour in self.contour.list_contour_types():
					_,frame_points = self.contour.get_frame_points(contour,
																   frame_uid)
					for c_point in frame_points:
						out.write('{0:.3f}\t{1:.3f}\t{2:.3f}\t'.format(
							c_point.coordinates[0],c_point.coordinates[1],
							c_point.coordinates[2])
							+ '{0}\t'.format( contour )
							+ '{0}\t{1}\t{2}'.format(
							self.contour.frame[frame_uid].image_id,
							c_point.weight,
							self.contour.frame[frame_uid].time_frame) +
								  '\n')
		out.close()

	def export_dicom_metadata(self, file_name, time_frame = None):


		if time_frame is None:
			time_frame = self.contour.list_time_frames()
		else:
			time_frame = time_frame

		if np.isscalar(time_frame):
			time_frame = [time_frame]

		if not os.path.exists(os.path.dirname(file_name)):
			os.makedirs(os.path.dirname(file_name))

		out = open(file_name,'w')

		for phase in time_frame:
			for frame_uid in self.contour.list_frame_uids_at_timeframe(phase):
				position = self.contour.get_frame_position(frame_uid)
				orientation = self.contour.get_frame_orientation(frame_uid)
				spacing = self.contour.get_frame_pixel_spacing(frame_uid)
				frame_id = self.contour.frame[frame_uid].image_id
				out.write(frame_uid + '\tframeID:\t{0}\ttimeFrame\t{1}'.format(
					frame_id,phase )+
						  '\tImagePositionPatient\t{0:.3f}\t{1:.3f}\t{'
									  '2:.3f}\t'.format(
							 position[0], position[1], position[2]) +
						  'ImageOrientationPatient\t{0:.3f}\t{1:.3f}\t{' \
						  '2:.3f}\t{3:.3f}\t{4:.3f}\t'
						'{5:.3f}\t'.format(orientation[0],orientation[ 1],
						orientation[2], orientation[3],orientation[4],
						orientation[5]) + 'PixelSpacing\t{0:.3f}\t{' \
										  '1:.3f}'.format(
						spacing[0],spacing[1])+'\n')


