# Author : Simon BERNARD
#
# Note that somes of these functions require a connection to cytomine beforehand :
#
# host="https://research.cytomine.be/"
# public_key=
# private_key=
#
# cytomine = reg.Cytomine.connect(host, public_key, private_key)
# print(cytomine.current_user)

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstanceCollection,ImageInstance, ProjectCollection, AnnotationLinkCollection, Annotation, AnnotationGroupCollection
from cytomine.utilities.annotations import get_annotations
import logging

from shapely import wkt
from shapely.affinity import affine_transform, scale, translate
from shapely.geometry import Polygon, Point

import numpy as np
import cv2

from PIL import Image

import matplotlib.pyplot as plt

import glob
import os
from math import floor, ceil

import registration as reg
import rasterio
import rasterio.features
import shapely
from affine import Affine


"""
Perform a single step registration by feature detection (using ORB and Hamming matching)

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_move_PIL : 		PIL image of the moving image
	img_static_PIL :	PIL image of the fixed image
	orig_move_size :	original size of the moving image (before fetching)
	orig_static_size :	original size of the fixed image (before fetching)
	plot_fig :			True|False -> plot the resulting registration

outputs :
	affine_t :	the shapely affine transform [m_00, m_01, m_10, m_11, m_02, m_12],
				the transform is in OpenCV coordinates

"""
def register_annot_v1_short(annot_move_wkt, img_move_PIL, img_static_PIL, orig_move_size, orig_static_size, init_path_img, plot_fig=False):

	## GET TRANSFORMATION
	img_move_cv = cv2.cvtColor(np.asarray(img_move_PIL), cv2.COLOR_RGB2GRAY)
	img_static_cv = cv2.cvtColor(np.asarray(img_static_PIL), cv2.COLOR_RGB2GRAY)

	# create ORB detector with 5000 features
	orb_detector = cv2.ORB_create(5000)

	# find keypoints and descriptors
	keyp_move, descr_move = orb_detector.detectAndCompute(img_move_cv, None)
	keyp_static, descr_static = orb_detector.detectAndCompute(img_static_cv, None)

	if descr_move is None or descr_static is None:
		return None


	# match features between the two images
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = matcher.match(descr_move, descr_static)

	# keep best features
	top_perc_kept = 0.9
	matches.sort(key = lambda x: x.distance)
	matches = matches[:int(len(matches)*top_perc_kept)]
	no_of_matches = len(matches)

	if no_of_matches == 0:
		return None

	pts_move = np.zeros((no_of_matches, 2))
	pts_static = np.zeros((no_of_matches, 2))

	for i in range(len(matches)):
		pts_move[i, :] = keyp_move[matches[i].queryIdx].pt
		pts_static[i, :] = keyp_static[matches[i].trainIdx].pt

	# find affine transform
	trans_aff, mask = cv2.estimateAffinePartial2D(pts_move, pts_static)

	#trans_aff, mask = cv2.estimateAffine2D(pts_move, pts_static)


	if trans_aff is None:
		return None

	# shapely transform
	affine_t = [trans_aff[0,0],trans_aff[0,1], trans_aff[1,0], trans_aff[1,1], trans_aff[0,2], trans_aff[1,2]]

	return affine_t

"""
Apply shapely transform to a Polygon annotation,
the transform was computed in OpenCV coordinate in the fetched resolution

inputs :
	annot :				Polygon of the annotation in the moving image (cartesian coordinate in moving image)
	affine_t :			the shapely affine transform [m_00, m_01, m_10, m_11, m_02, m_12],
						the transform is in OpenCV coordinates
	orig_move_size :	original size of the moving image (before fetching)
	orig_static_size :	original size of the fixed image (before fetching)
	PIL_move_size :		fetched size of the moving image
	PIL_static_size :	fetched size of the fixed image

output :
	the registered Polygon annotation (cartesian coordinate in fixed image)

"""
def apply_trans_annot(annot, affine_t, orig_move_size, orig_static_size, PIL_move_size, PIL_static_size):
	# transfom annotation
	# scale down to image PIL dump
	annot = reg.rescale_annot(annot, orig_move_size, PIL_move_size)
	# change to cv coordinate
	annot = affine_transform(annot, [1, 0, 0, -1, 0, PIL_move_size[1]])
	# apply affine transformation
	annot = affine_transform(annot, affine_t)
	# go back to cartesian coordinate
	annot = affine_transform(annot, [1, 0, 0, -1, 0, PIL_static_size[1]])
	# rescale up to original size
	annot = reg.rescale_annot(annot, PIL_static_size, orig_static_size)

	return annot

"""
Build the transformation to create the image aligned for the dataset

inputs :
	window_move :		[left, bottom, width, height] of the window fetched from the moving image
	window_final_size : size of the window [width, height] after alignement
	frame_center_init :	initial center of interest in the moving window
	orig_move_size :	original size of the moving image (before fetching)
	orig_static_size :	original size of the fixed image (before fetching)
	PIL_move_size : 	fetched size of the moving image
	PIL_static_size : 	fetched size of the fixed image
	affine_t : 			the shapely affine transform [m_00, m_01, m_10, m_11, m_02, m_12],
						the transform is in OpenCV coordinates
	max_dump_size :		max dump size used for the windows (and final image size),
						window_move is supposed initially fetched at twice this value before alignement

outputs :
	trans_tot_cv			: the transform for the image consider an OpenCV image of max_size "max_dump_size*2"
								with its relative origin at (0,0)
	trans_tot_annot_move	: the transform for the annotation start from the original cartesian system in the moving image
	trans_tot_annot_static	: the transform for the annotation start from the original cartesian system in the static image

"""
def build_trans_mats(window_move, window_final_size, frame_center_init,
					orig_move_size, orig_static_size, PIL_move_size, PIL_static_size,
					affine_t, max_dump_size):

	# trans_tot_cv is to make the transformation from OpenCV coordinate
	# trans_tot_annot is to make the transformation from the cartesian coordinate
	trans_tot_cv = np.eye(3)
	trans_tot_annot_move = np.eye(3)
	trans_tot_annot_static = np.eye(3)

	## go back to initial scale
	# note that the image of the annotation that will be moved is assumed to be downloaded
	# with a max_dump_size twice the original, then scaled down and cropped at the end to max_dump_size
	shrink_move_annot = max(1, max(window_move[2:])/(max_dump_size*2))
	rescale_mat_1 = np.array([[shrink_move_annot,0,0],
							[0,shrink_move_annot,0],
							[0,0,1]])

	trans_tot_cv = np.matmul(rescale_mat_1, trans_tot_cv)
	#trans_tot_annot_move = np.matmul(rescale_mat_1, trans_tot_annot_move)

	## translate to original place in image (different for cartesian and cv)
	translate_mat_1_cv = np.array([[1,0,window_move[0]],
									[0,1,orig_move_size[1]- window_move[1] - window_move[3]],
									[0,0,1]])
	translate_mat_1_annot = np.array([[1,0,window_move[0]],
									[0,1,window_move[1]],
									[0,0,1]])

	trans_tot_cv = np.matmul(translate_mat_1_cv, trans_tot_cv)
	#trans_tot_annot_move = np.matmul(translate_mat_1_annot, trans_tot_annot_move)

	## rescale to registration scale
	shrink_move_img = max(1, max(orig_move_size)/max_dump_size)
	rescale_to_reg_mat = np.array([[1/shrink_move_img,0,0],
									[0,1/shrink_move_img,0],
									[0,0,1]])

	trans_tot_cv = np.matmul(rescale_to_reg_mat, trans_tot_cv)
	trans_tot_annot_move = np.matmul(rescale_to_reg_mat, trans_tot_annot_move)

	## change to cv coord (for the cartesian case i.e. annotations)
	to_cv_mat = np.array([[1,0,0],
						[0,-1,PIL_move_size[1]],
						[0,0,1]])

	trans_tot_annot_move = np.matmul(to_cv_mat, trans_tot_annot_move)

	## affine registration matrix
	registration_mat = np.array([[affine_t[0], affine_t[1], affine_t[4]],
								[affine_t[2], affine_t[3], affine_t[5]],
								[0,0,1]])

	trans_tot_cv = np.matmul(registration_mat, trans_tot_cv)
	trans_tot_annot_move = np.matmul(registration_mat, trans_tot_annot_move)

	## go back to cartesian coord (for the cartesian case i.e. annotations)
	to_cart_mat = np.array([[1,0,0],
							[0,-1,PIL_static_size[1]],
							[0,0,1]])

	trans_tot_annot_move = np.matmul(to_cart_mat, trans_tot_annot_move)

	## rescale to original size
	shrink_static_img = max(1, max(orig_static_size)/max_dump_size)
	rescale_to_img_mat = np.array([[shrink_static_img,0,0],
									[0,shrink_static_img,0],
									[0,0,1]])

	trans_tot_cv = np.matmul(rescale_to_img_mat, trans_tot_cv)
	trans_tot_annot_move = np.matmul(rescale_to_img_mat, trans_tot_annot_move)

	## center to final frame
	frame_center_reg = apply_trans_annot(frame_center_init, affine_t, orig_move_size, orig_static_size, PIL_move_size, PIL_static_size)
	translate_mat_2_annot = np.array([[1,0,-frame_center_reg.x + window_final_size[0]/2],
									[0,1,-frame_center_reg.y + window_final_size[1]/2],
									[0,0,1]])

	frame_center_reg_cv = Point([frame_center_reg.x, orig_static_size[1] - frame_center_reg.y])
	translate_mat_2_cv = np.array([[1,0,-frame_center_reg_cv.x + window_final_size[0]/2],
									[0,1,-frame_center_reg_cv.y + window_final_size[1]/2],
									[0,0,1]])

	trans_tot_cv = np.matmul(translate_mat_2_cv, trans_tot_cv)
	trans_tot_annot_move = np.matmul(translate_mat_2_annot, trans_tot_annot_move)
	trans_tot_annot_static = np.matmul(translate_mat_2_annot, trans_tot_annot_static)

	## rescale to dump_size
	shrink_move_annot_small = max(1, max(window_final_size)/max_dump_size)
	rescale_to_dump_size = np.array([[1/shrink_move_annot_small,0,0],
									[0,1/shrink_move_annot_small,0],
									[0,0,1]])

	trans_tot_cv = np.matmul(rescale_to_dump_size, trans_tot_cv)
	trans_tot_annot_move = np.matmul(rescale_to_dump_size, trans_tot_annot_move)
	trans_tot_annot_static = np.matmul(rescale_to_dump_size, trans_tot_annot_static)

	return trans_tot_cv, trans_tot_annot_move, trans_tot_annot_static

"""
Build the 4 images required for the deep learning dataset from a given pair of annotation :
the two aligned images (with 1 step orb detector) and the masks of their annotations

inputs :
	annot_move_wkt :	Polygon annotation from the moving image
	img_inst_move :		ImageInstance of the moving image
	annot_static_wkt :	Polygon annotation from the fixed image
	img_inst_static :	ImageInstance of the fixed image
	init_path_imgs :	path of the folder containing the fetched image of the dataset
	dataset_folder :	folder where the images of the deep learning dataset should be placed
	name :				nickname of the pair of annotation
	max_dump_size :		max resolution for dumping, final resolution of the 4 images

"""
def build_sample_from_annot(annot_move_wkt, img_inst_move, annot_static_wkt, img_inst_static,
							init_path_imgs, dataset_folder="tmp/", name="",
							max_dump_size=512):

	## Name for the 4 resulting images (+removed original dump)
	# original dump (removed)
	name_window_move_dataset = name+"window_move_dataset"
	# registered/scaled/cropped dump
	name_window_moved_dataset = name+"window_moved_dataset"
	# dump from the static image
	name_window_static_dataset = name+"window_static_dataset"
	# binary mask of the registered annotation
	name_window_moved_mask_dataset = name+"window_moved_mask_dataset"
	# binary mask of the static annotation
	name_window_static_mask_dataset = name+"window_static_mask_dataset"

	# load PIL image
	img_move_PIL = Image.open(init_path_imgs+img_inst_move.originalFilename+".jpg")
	img_static_PIL = Image.open(init_path_imgs+img_inst_static.originalFilename+".jpg")

	# store item sizes
	orig_move_size = [img_inst_move.width, img_inst_move.height]
	orig_static_size = [img_inst_static.width, img_inst_static.height]
	PIL_move_size = [img_move_PIL.width, img_move_PIL.height]
	PIL_static_size = [img_static_PIL.width, img_static_PIL.height]

	# perform registration
	affine_t = register_annot_v1_short(annot_move_wkt,
								   img_move_PIL,
								   img_static_PIL,
								   orig_move_size,
								   orig_static_size,
								   init_path_imgs,
								   plot_fig=False)

	if affine_t is None:
		print("Registration failed, abort sample construction")

		return

	# overhead in percentage (of the annotation) of the area around the annotation to keep
	incr_look = 1
	# if initial annot is too small, take a minimum percentage of the image
	min_img_perc = 0.1
	# overhead that is supposed to be discarded (but may appear trough rotation)
	# for a square, sqrt(2)/2 (< 1.5) times the image before rotation is theoriticaly enough
	incr_rot_scale = 0.5

	# compute window to dump initial dimension (supposed entierly in the image)
	(minx_move, miny_move, maxx_move, maxy_move) = annot_move_wkt.bounds
	window_move_width = maxx_move - minx_move
	window_move_height = maxy_move - miny_move

	# keep the center of the area we are interested in
	frame_center_init = Point([minx_move + window_move_width/2, miny_move + window_move_height/2])

	# build the final window to dump (left, bottom, width, height)
	window_move = [0,0,0,0]

	# enforce square patch
	max_dim = max(window_move_width, window_move_height)
	# make sure at least "min_img_perc" is taken from the image (in the smallest diemnsion)
	if ((1+incr_look)*max_dim < min_img_perc*img_inst_move.width or
		(1+incr_look)*max_dim < min_img_perc*img_inst_move.height):
		max_dim = min_img_perc*max(img_inst_move.width, img_inst_move.height)/(1+incr_look)

	# keep size of the final window (without rotation overhead)
	window_final_size = [ceil((1+incr_look)*max_dim), ceil((1+incr_look)*max_dim)]

	# add surrounding area at "incr_look" percentage of the annotation
	# avoid out-of-image (negative)
	window_move[:2] = [max(0, floor(minx_move - (incr_look/2)*max_dim)),
						max(0, floor(miny_move - (incr_look/2)*max_dim))]
	# avoid out-of-image (greater than image size)
	window_move[2:4] = [min(ceil((1+incr_look)*max_dim), img_inst_move.width-window_move[0]),
						min(ceil((1+incr_look)*max_dim), img_inst_move.height-window_move[1])]

	# add a small overhead in case of rotation ("incr_rot_scale" percentage of the window)
	window_move[:2] = [max(0, floor(window_move[0] - (incr_rot_scale/2)*window_move[2])),
						max(0, floor(window_move[1] - (incr_rot_scale/2)*window_move[3]))]
	window_move[2:4] = [min(ceil((1+incr_rot_scale)*window_move[2]), img_inst_move.width-window_move[0]),
						min(ceil((1+incr_rot_scale)*window_move[3]), img_inst_move.height-window_move[1])]

	# DOWNLOAD dump of the window with twice the max_dump_size
	# (to keep enough detail after registration)
	img_inst_move.window(window_move[0],
						img_inst_move.height - window_move[1] - window_move[3],
						window_move[2],
						window_move[3],
						dest_pattern=dataset_folder+name_window_move_dataset+".png",
						max_size=max_dump_size*2)

	# build affine transformation matrices for the dump image and the annotation
	trans_cv, trans_annot_move, trans_annot_static = build_trans_mats(window_move, window_final_size, frame_center_init,
																			orig_move_size, orig_static_size, PIL_move_size, PIL_static_size,
																			affine_t, max_dump_size)

	# get annotation affine_t_move list for shapely
	affine_t_annot_move = [trans_annot_move[0,0],trans_annot_move[0,1], trans_annot_move[1,0],
							trans_annot_move[1,1], trans_annot_move[0,2], trans_annot_move[1,2]]

	# get annotation affine_t_static list for shapely
	affine_t_annot_static = [trans_annot_static[0,0],trans_annot_static[0,1], trans_annot_static[1,0],
							trans_annot_static[1,1], trans_annot_static[0,2], trans_annot_static[1,2]]


	# abort if window messed up such that no windows has been dumped
	try:
		# open dumped window
		window_PIL = Image.open(dataset_folder+name_window_move_dataset+".png")
	except FileNotFoundError:
		print("Moving window unable to dump, abort sample construction")
		return

	# perform registration
	trans_window = cv2.warpPerspective(cv2.cvtColor(np.asarray(window_PIL), cv2.COLOR_RGB2BGR),
									  trans_cv, (max_dump_size,max_dump_size))
	trans_window = cv2.cvtColor(trans_window, cv2.COLOR_BGR2RGB)
	trans_window_pil = Image.fromarray(trans_window)

	# save transformed window
	trans_window_pil.save(dataset_folder+name_window_moved_dataset+".png")
	# remove previous one
	os.remove(dataset_folder+name_window_move_dataset+".png")

	# get window for static image
	frame_center_reg = apply_trans_annot(frame_center_init, affine_t, orig_move_size, orig_static_size, PIL_move_size, PIL_static_size)
	window_static = [int(frame_center_reg.x - window_final_size[0]/2),
					int(frame_center_reg.y - window_final_size[1]/2),
					window_final_size[0],
					window_final_size[1]]

	# border correction (left, bottom, right, top)
	border_corr = [0,0,0,0]

	# avoid out-of-image (negative)
	border_corr[:2] = [max(0, -floor(window_static[0])),
					max(0, -floor(window_static[1]))]
	window_static[:2] = [max(0, floor(window_static[0])),
						max(0, floor(window_static[1]))]

	window_static[2:4] = [window_static[2]-border_corr[0],
							window_static[3]-border_corr[1]]
	# avoid out-of-image (greater than image size)
	border_corr[2:4] = [max(0, window_static[2]-img_inst_move.width-window_static[0]),
						max(0, window_static[3]-img_inst_move.height-window_static[1])]
	window_static[2:4] = [min(window_static[2], img_inst_move.width-window_static[0]),
							min(window_static[3], img_inst_move.height-window_static[1])]

	# turn border correction to percentage of the window
	border_corr[0] /= window_static[2]
	border_corr[1] /= window_static[3]
	border_corr[2] /= window_static[2]
	border_corr[3] /= window_static[3]



	# DOWNLOAD dump of the window in static image with max_dump_size
	img_inst_static.window(window_static[0],
							img_inst_static.height - window_static[1] - window_static[3],
							window_static[2],
							window_static[3],
							dest_pattern=dataset_folder+name_window_static_dataset+".png", max_size=max_dump_size)



	# abort if window messed up such that no windows has been dumped
	try:
		# open dumped window
		window_static_PIL = Image.open(dataset_folder+name_window_static_dataset+".png")
	except FileNotFoundError:
		print("Static window unable to dump, abort sample construction")
		return

	# if border erosion somewhere
	if np.array(border_corr).any():
		# get border size
		border_corr[0] *= window_static_PIL.width
		border_corr[1] *= window_static_PIL.height
		border_corr[2] *= window_static_PIL.width
		border_corr[3] *= window_static_PIL.height

		# make sure static dump is of (max_dump_size, max_dump_size) dimensions and well aligned
		new_dim = (window_static_PIL.width+int(border_corr[0]+border_corr[2]),
					window_static_PIL.height+int(border_corr[1]+border_corr[3]))
		window_static_PIL_new = Image.new(window_static_PIL.mode, new_dim, 0)
		window_static_PIL_new.paste(window_static_PIL, (int(border_corr[0]), int(border_corr[3])))
		window_static_PIL_new = window_static_PIL_new.resize((max_dump_size, max_dump_size), Image.ANTIALIAS)

		window_static_PIL.close()
		window_static_PIL_new.save(dataset_folder+name_window_static_dataset+".png")
		window_static_PIL_new.close()

	else:
		window_static_PIL.close()

	# register/scale/crop annotation (move)
	annot_move_wkt_dataset = affine_transform(annot_move_wkt, affine_t_annot_move)

	# turn to cv coordinate
	annot_move_wkt_dataset = affine_transform(annot_move_wkt_dataset, [1, 0, 0, -1, 0, max_dump_size])

	# create binary mask (moved annotation)
	mask_moved = rasterio.features.rasterize([annot_move_wkt_dataset], out_shape=(max_dump_size,max_dump_size),default_value=255)

	# save mask (moved annotation)
	cv2.imwrite(dataset_folder+name_window_moved_mask_dataset+".png", mask_moved)

	# scale/crop annotation (static)
	annot_static_wkt_dataset = affine_transform(annot_static_wkt, affine_t_annot_static)

	# turn to cv coordinate
	annot_static_wkt_dataset = affine_transform(annot_static_wkt_dataset, [1, 0, 0, -1, 0, max_dump_size])

	# create binary mask (static annotation)
	mask_static = rasterio.features.rasterize([annot_static_wkt_dataset], out_shape=(max_dump_size,max_dump_size),default_value=255)

	# save mask (static annotation)
	cv2.imwrite(dataset_folder+name_window_static_mask_dataset+".png", mask_static)


"""
Build nickname as follow :
<img_name1>(<annot_id_1>)+<img_name2>(<annot_id_2>)

inputs :
	img_inst_move : 	ImageInstance of the moving image
	annot_move :		Annotation in the moving image
	img_inst_static :	ImageInstance of the fixed image
	annot_static :		Annotation in the fixed image

"""
def build_name(img_inst_move, annot_move, img_inst_static, annot_static):

	name_1 = img_inst_move.originalFilename
	pos_dot = name_1.find('.')
	if pos_dot != -1:
		name_1 = name_1[:pos_dot]

	name_2 = img_inst_static.originalFilename
	pos_dot = name_2.find('.')
	if pos_dot != -1:
		name_2 = name_2[:pos_dot]

	return name_1+"("+str(annot_move.id)+")+"+name_2+"("+str(annot_static.id)+")"

"""
Build the 4 images required for the deep learning dataset for ALL pairs of annotation :
the two aligned images (with 1 step orb detector) and the masks of their annotations

inputs :
	annot_group_coll : 	AnnotationGroupCollection of the different annotation group, can be fetched with :
						AnnotationGroupCollection().fetch_with_filter("project", id_project)
						where <id_project> is the id of the project to test
	init_path_imgs :	path of the folder containing the fetched image of the dataset
	dataset_folder :	folder where the images of the deep learning dataset should be placed
	override :			True|False -> True if the 4 images have to be rebuilt even if already present in the dataset_folder

"""
def build_all(annot_group_coll, init_path_imgs="fetched_imgs/", dataset_folder="tmp/", override=False):

	i=0

	for annot_group in annot_group_coll:

		print("\tAnnotation group : {}".format(annot_group.id))
		annot_link_coll = AnnotationLinkCollection().fetch_with_filter("annotationgroup", annot_group.id)

		if len(annot_link_coll) <= 1:
			continue

		# list annotations that will be moved onto other one
		for annot_link_move in annot_link_coll:
			# annotation and image cytomine objects (move)
			annot_move = Annotation(id=annot_link_move.annotationIdent, showWKT=True, showMeta=True, showGIS=True, showLink=True).fetch()
			annot_move_wkt = reg.multipol_handler(wkt.loads(annot_move.location))
			img_inst_move = ImageInstance(id=annot_move.image).fetch()

			# list true annotation to assess performances
			for annot_link_static in annot_link_coll:
				# do not move an annotation on themselves
				if annot_link_move.id == annot_link_static.id:
					continue

				# annotation and image cytomine objects (static)
				annot_static = Annotation(id=annot_link_static.annotationIdent, showWKT=True, showMeta=True, showGIS=True, showLink=True).fetch()
				annot_static_wkt = reg.multipol_handler(wkt.loads(annot_static.location))
				img_inst_static = ImageInstance(id=annot_static.image).fetch()

				print("\t\t{} (from {}) MAPPED TO {} (from {})".format(annot_move.id, img_inst_move.id, annot_static.id, img_inst_static.id))

				# build unique name for the pairing
				name_prefix = build_name(img_inst_move, annot_move, img_inst_static, annot_static)

				i+=1
				print(i)

				# do not rebuild existing file
				if (not override) and os.path.isfile(dataset_folder+name_prefix+"+window_static_mask_dataset.png"):
					print("\""+name_prefix+"\" already exist")
					continue

				build_sample_from_annot(annot_move_wkt, img_inst_move, annot_static_wkt, img_inst_static,
                            			init_path_imgs, dataset_folder=dataset_folder, name=name_prefix+"+",
                            			max_dump_size=512)

