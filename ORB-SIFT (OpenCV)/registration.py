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
from shapely.geometry import Polygon

from math import floor, ceil

import numpy as np
import cv2

from PIL import Image, ImageOps
import os

import matplotlib.pyplot as plt

"""
plot the given polygons on a single graph

inputs :

	pol_list :				a list of shapely Polygons to plot
	pol_names :				names of the Polygons for the legend
	plot_cent :				True|False -> plot the centroid of the annotation
	plot_leg :				True|False -> plot the legend
	plot_save_path :		path (including name) to save the image (image not save if set to None)

"""
def plot_annots(pol_list, pol_names=None, plot_cent=False, plot_leg=True, plot_save_path=None):
	if pol_names == None or len(pol_list) != len(pol_names):
		pol_names = ["image {}".format(i+1) for i in range(len(pol_list))]

	fig = plt.figure()

	ax = plt.gca()

	for idx, wkt_annot in enumerate(pol_list):
		color = next(ax._get_lines.prop_cycler)['color']

		if wkt_annot.geom_type == "Point":
			plt.plot(wkt_annot.x, wkt_annot.y, marker='o', color=color)

		else:
			x,y = wkt_annot.exterior.xy
			plt.plot(x,y, label=pol_names[idx], color=color)

			if plot_cent:
				centroid = wkt_annot.centroid
				plt.plot(centroid.x, centroid.y, label=pol_names[idx], marker='o', color=color)

	plt.axis("equal")
	if plot_leg:
		plt.legend()
	plt.show()

	if plot_save_path is not None:
		fig.savefig(plot_save_path)

	plt.clf()

"""
given the minimum and maximum x/y values, return the corresponding Polygon of the bounding box

inputs :
	bounds :	list/tuple of (minx, miny, maxx, maxy)

outputs :
	shapely Polygon

"""
def frame(bounds):
	(minx, miny, maxx, maxy) = bounds
	return Polygon([[minx, miny],[maxx, miny],[maxx, maxy],[minx, maxy],[minx, miny]])

"""
Given a Polygon/Mutlipolygon, return the Polygon with the greatest area (identity if already Polygon)

inputs :
	my_pol :	Polygon/Mutlipolygon to treat

outputs :
	Polygon with greatest area

"""
def multipol_handler(my_pol):
	if my_pol.geom_type == 'MultiPolygon':
		my_pol = max(my_pol, key=lambda a: a.area)

	return my_pol


"""
Rescale an annotation given the original dimension of the image and the current one.
Used when image fetched in lower resolution.

inputs :
	annot_pol :		the Polygon annotation to rescale
	dim_orig :		the original dimension of the image [width, height]
	dim_rescaled :		the current dimension of the image [width, height]

"""
def rescale_annot(annot_pol, dim_orig, dim_rescaled):
	width_ratio = dim_orig[0]/dim_rescaled[0]
	height_ratio = dim_orig[1]/dim_rescaled[1]

	return scale(annot_pol, xfact=1/width_ratio, yfact=1/height_ratio, origin=(0,0))

"""
Given 2 Polygons (of annotations), returns the metrics associated (IoU and RCM)

inputs :
	poly1 :		first Polygon of annotation
	poly2 :		second Polygon of annotation
	ref_size :	reference size for the RCM

outputs :
	dist_c_rel : 	the RCM, the distance between the centroid divided by the reference size given (default is 1)
	iou :			the IoU, intersection over union, of the two Polygons
"""
def metrics(poly1, poly2, ref_size=1):
	# distance of centroids over a reference (eg : diagonal of the image)
	c1 = poly1.centroid
	c2 = poly2.centroid
	dist_c_rel = c1.distance(c2)/ref_size

	# IoU
	iou = poly1.intersection(poly2).area/ poly1.union(poly2).area

	return dist_c_rel, iou

"""
Extract the stain used from the name of the image
<image type>-<number>-<stains>.<extension>

inputs :
	name :	full image name

outputs :
	the string containing the stain(s) used

"""
def get_stain(name):
	return name.rpartition('-')[-1].partition('.')[0]

"""
Extract the different stains used from the name of the stain extracted from get_stain()
<stain1>_<stain2>_<stain3>_...

inputs :
	name :	full stain name

outputs :
	stain_list :	the list of stains used

"""
def split_stain(stain_name):
	stain_list = []

	next_stain = stain_name.find('_')
	while next_stain != -1:
		stain_list.append(stain_name[:next_stain])
		stain_name = stain_name[next_stain+1:]
		next_stain = stain_name.find('_')

	stain_list.append(stain_name)

	return stain_list


"""
Given several dictionnaries (for full stain, split stain, ...),
complete/add the entries for the different metrics for the key "name" (in place)

input :
	dict_occ :		dictionnary of the occurences
	dict_IoU_base :	dictionnary of the IoUs (without registration)
	dict_err_base :	dictionnary of the RCMs (without registration)
	dict_IoU_reg :	dictionnary of the IoUs (with registration)
	dict_err_reg :	dictionnary of the RCMs (with registration)
	name :			key for the dictionnaries
	IoU_base :		IoU without registration to include in dictionnary "dict_IoU_base"
	cent_err_base :	RCM without registration to include in dictionnary "dict_err_base"
	IoU_reg :		IoU with registration to include in dictionnary "dict_IoU_reg"
	cent_err_reg :	RCM with registration to include in dictionnary "dict_err_reg"
"""
def add_result_dicts(dict_occ, dict_IoU_base, dict_err_base, dict_IoU_reg, dict_err_reg,
					name, IoU_base, cent_err_base, IoU_reg, cent_err_reg):

	if name in dict_occ:
		dict_occ[name] += 1
		dict_IoU_base[name] += IoU_base
		dict_err_base[name] += cent_err_base
		dict_IoU_reg[name] += IoU_reg
		dict_err_reg[name] += cent_err_reg

	# add stain if never seen
	else:
		dict_occ[name] = 1
		dict_IoU_base[name] = IoU_base
		dict_err_base[name] = cent_err_base
		dict_IoU_reg[name] = IoU_reg
		dict_err_reg[name] = cent_err_reg

"""
Compute the average of the dictionnaries for IoU and RCM with the number of occurences for each entries (in place)

input :
	dict_occ :	dictionnary of the occurences
	dict_IoU :	dictionnary of the IoUs
	dict_err :	dictionnary of the RCMs

"""
def get_means_dicts(dict_occ, dict_IoU, dict_err):
	for key in dict_occ:
		dict_IoU[key] /= dict_occ[key]
		dict_err[key] /= dict_occ[key]

"""
Print the dictionnaries
input :
	dict_occ :		dictionnary of the occurences
	dict_n_annot :	dictionnary of the number of unique annotation for each key/stain
	dict_imgs :		dictionnary of the images for each key/stain
	dict_fails :	dictionnary of the number of fail (during registration) for each key/stain
	dict_IoU_base :	dictionnary of the IoUs (without registration)
	dict_err_base :	dictionnary of the RCMs (without registration)
	dict_IoU_reg :	dictionnary of the IoUs (with registration)
	dict_err_reg :	dictionnary of the RCMs (with registration)
"""
def print_dicts(dict_occ, dict_n_annot, dict_imgs, dict_fails, dict_IoU_base, dict_err_base, dict_IoU_reg, dict_err_reg):
	print("stain\nn_annot\nn_occ\nn_img\nimages\nIoU_base\ncent_err_base\nIoU_reg\ncent_err_reg\nfails\n")
	for key in dict_occ:
		print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(key, dict_n_annot[key], dict_occ[key],
																len(dict_imgs[key]), dict_imgs[key],
																dict_IoU_base[key], dict_err_base[key],
																dict_IoU_reg[key], dict_err_reg[key], dict_fails[key]))

"""
Perform a single step registration by feature detection,
wrapper for the register_annot_v1() using Cytomine ImageInstance as input instead

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_inst_move : 	ImageInstance of the moving image
	img_inst_static :	ImageInstance of the fixed image
	init_path_img :		path of the folder containing the fetched image of the dataset
	orb :				True|False -> use the Orb detector (with Hamming matching) if True (default),
										use SIFT (with L2 matching) otherwise
	full_aff :			True|False -> try to find a full affine transform if True,
										find an affine without shearing otherwise (default)
	plot_fig :			True|False -> plot the resulting registration

outputs :
	new_annot :		the registered Polygon annotation

"""
def register_annot_v1_cyt(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, orb=True, full_aff=False, plot_fig=False):
	img_move_PIL = Image.open(init_path_img+img_inst_move.originalFilename+".jpg")

	img_static_PIL = Image.open(init_path_img+img_inst_static.originalFilename+".jpg")


	new_annot =  register_annot_v1(annot_move_wkt, img_move_PIL, img_static_PIL,
										[img_inst_move.width, img_inst_move.height],
										[img_inst_static.width, img_inst_static.height],
										orb=orb, full_aff=full_aff, plot_fig=plot_fig)

	img_move_PIL.close()
	img_static_PIL.close()

	return new_annot

"""
Perform a single step registration by feature detection

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_move_PIL : 		PIL image of the moving image
	img_static_PIL :	PIL image of the fixed image
	orig_move_size :	original size of the moving image (before fetching)
	orig_static_size :	original size of the fixed image (before fetching)
	orb :				True|False -> use the Orb detector (with Hamming matching) if True (default),
										use SIFT (with L2 matching) otherwise
	full_aff :			True|False -> try to find a full affine transform if True,
										find an affine without shearing otherwise (default)
	plot_fig :			True|False -> plot the resulting registration

outputs :
	the registered Polygon annotation (or None if something went wrong)

"""
def register_annot_v1(annot_move_wkt, img_move_PIL, img_static_PIL, orig_move_size, orig_static_size, orb=True, full_aff=False, plot_fig=False):

	# turn images to OpenCV images
	img_move_cv = cv2.cvtColor(np.asarray(img_move_PIL), cv2.COLOR_RGB2GRAY)
	img_static_cv = cv2.cvtColor(np.asarray(img_static_PIL), cv2.COLOR_RGB2GRAY)

	if orb:
		# create ORB detector with 5000 features
		detector = cv2.ORB_create(5000)
	else:
		detector = cv2.SIFT_create()

	# find keypoints and descriptors
	keyp_move, descr_move = detector.detectAndCompute(img_move_cv, None)
	keyp_static, descr_static = detector.detectAndCompute(img_static_cv, None)

	if descr_move is None or descr_static is None:
		return None


	# match features between the two images
	if orb:
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	else:
		matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)

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

	# find partial affine transform (default parameter)
	if not full_aff:
		trans_aff, mask = cv2.estimateAffinePartial2D(pts_move, pts_static)

	# find full affine transform (if full_aff set to true)
	else:
		trans_aff, mask = cv2.estimateAffine2D(pts_move, pts_static)

	if (trans_aff is None) or (np.isnan(trans_aff).any()):
		return None



	### plot transformation
	if plot_fig:
		affine_matrix = np.append(trans_aff, np.array([[0,0,1]]), axis=0)
		transformed_img = cv2.warpPerspective(cv2.cvtColor(np.asarray(img_move_PIL), cv2.COLOR_RGB2BGR), affine_matrix, (img_static_cv.shape[1] , img_static_cv.shape[0]))

		transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
		transformed_img_pil = Image.fromarray(transformed_img)

		display(img_static_PIL)
		display(transformed_img_pil)

		transformed_img_pil.close()

	# shapely transform
	affine_t = [trans_aff[0,0],trans_aff[0,1], trans_aff[1,0], trans_aff[1,1], trans_aff[0,2], trans_aff[1,2]]

	# transfom annotation
	# scale down to image PIL dump
	annot_move_wkt = rescale_annot(annot_move_wkt, orig_move_size, [img_move_PIL.width, img_move_PIL.height])
	# change to cv coordinate
	annot_move_wkt = affine_transform(annot_move_wkt, [1, 0, 0, -1, 0, img_move_PIL.height])
	# apply affine transformation
	annot_move_wkt = affine_transform(annot_move_wkt, affine_t)
	# go back to cartesian coordinate
	annot_move_wkt = affine_transform(annot_move_wkt, [1, 0, 0, -1, 0, img_static_PIL.height])
	# rescale up to original size
	annot_move_wkt = rescale_annot(annot_move_wkt, [img_static_PIL.width, img_static_PIL.height], orig_static_size)

	return annot_move_wkt


"""
Perform registration for all pairs and summarize the results for the IoU and RCM metrics
REQUIRE CYTOMINE CONNECTION

inputs :
	annot_group_coll : 	AnnotationGroupCollection of the different annotation group, can be fetched with :
						AnnotationGroupCollection().fetch_with_filter("project", id_project)
						where <id_project> is the id of the project to test
	init_path_imgs :	path of the folder containing the fetched image of the dataset
	version : 			string describing the version to use
						-> "single" is the single step registration
						-> "pattern" is the two steps pattern matching registration
						-> "large" is the two steps registration with the large window for both images
						-> "validation" perform both one step and large two steps,
							but keep the two step only if it overlap the one step
							with at least "thresh" of IoU (see parameter later)
	orb :				True|False -> use the Orb detector (with Hamming matching) if True (default),
										use SIFT (with L2 matching) otherwise
	full_aff :			True|False -> try to find a full affine transform if True (ONLY AVAILABLE FOR V1),
										find an affine without shearing otherwise (default)
	thresh :			[0,1] -> parameter for the "validation" version, determine how much
						the one step and two steps should overlap (in IoU) to accept the two steps



"""
def metrics_reg_all(annot_group_coll, init_path_imgs="dl-file/526102245/reduced_img/", version="single", orb=True, full_aff=False, thresh=0.5):

	annot_grp_ids = []
	annot_grp_n_pairs = []
	annot_grp_fails = []
	metrics_base = []
	metrics_reg = []
	n_pair = 0
	mean_centroid_err_base = 0
	mean_IoU_base = 0
	mean_centroid_err_reg = 0
	mean_IoU_reg = 0

	fails = 0

	stain_dict_occ = {}
	stain_dict_nbr_ann = {}
	stain_dict_imgs = {}
	stain_dict_fails = {}
	stain_dict_IoU_base = {}
	stain_dict_centroid_err_base = {}
	stain_dict_IoU_reg = {}
	stain_dict_centroid_err_reg = {}

	stain_dict_split_occ = {}
	stain_dict_split_nbr_ann = {}
	stain_dict_split_imgs = {}
	stain_dict_split_fails = {}
	stain_dict_split_IoU_base = {}
	stain_dict_split_centroid_err_base = {}
	stain_dict_split_IoU_reg = {}
	stain_dict_split_centroid_err_reg = {}

	for annot_group in annot_group_coll:

		print("\tAnnotation group : {}".format(annot_group.id))
		annot_link_coll = AnnotationLinkCollection().fetch_with_filter("annotationgroup", annot_group.id)

		# need at least 2 annotations in the group
		if len(annot_link_coll) <= 1:
			continue

		# metrics per group
		annot_grp_ids.append(annot_group.id)
		annot_grp_n_pairs.append(0)
		annot_grp_fails.append(0)
		metrics_base.append([0,0])
		metrics_reg.append([0,0])

		# list annotations that will be moved onto other one
		for annot_link_move in annot_link_coll:


			# annotation and image cytomine objects (move)
			annot_move = Annotation(id=annot_link_move.annotationIdent, showWKT=True, showMeta=True, showGIS=True, showLink=True).fetch()
			annot_move_wkt = multipol_handler(wkt.loads(annot_move.location))
			img_inst_move = ImageInstance(id=annot_move.image).fetch()

			# extract stains
			stain_move = get_stain(img_inst_move.instanceFilename)
			stain_move_list = split_stain(stain_move)

			# add stain entry
			# full stain name
			if stain_move in stain_dict_nbr_ann:
				stain_dict_nbr_ann[stain_move] += 1
				stain_dict_imgs[stain_move].add(img_inst_move.originalFilename)
			else:
				stain_dict_nbr_ann[stain_move] = 1
				stain_dict_imgs[stain_move] = {img_inst_move.originalFilename}
				stain_dict_fails[stain_move] = 0

			# stains split list
			for stain in stain_move_list:
				if stain in stain_dict_split_nbr_ann:
					stain_dict_split_nbr_ann[stain] += 1
					stain_dict_split_imgs[stain].add(img_inst_move.originalFilename)
				else:
					stain_dict_split_nbr_ann[stain] = 1
					stain_dict_split_imgs[stain] = {img_inst_move.originalFilename}
					stain_dict_split_fails[stain] = 0

			# list true annotation to assess performances
			for annot_link_static in annot_link_coll:
				# do not move an annotation on themselves
				if annot_link_move.id == annot_link_static.id:
					continue

				# annotation and image cytomine objects (static)
				annot_static = Annotation(id=annot_link_static.annotationIdent, showWKT=True, showMeta=True, showGIS=True, showLink=True).fetch()
				annot_static_wkt = multipol_handler(wkt.loads(annot_static.location))
				img_inst_static = ImageInstance(id=annot_static.image).fetch()

				print("\t\t{} (from {}) MAPPED TO {} (from {})".format(annot_move.id, img_inst_move.id, annot_static.id, img_inst_static.id))


				# REGISTRATION
				if version == "single":
					annot_reg_wkt = register_annot_v1_cyt(annot_move_wkt, img_inst_move, img_inst_static, init_path_imgs, orb=orb, full_aff=full_aff)
				elif version == "pattern":
					annot_reg_wkt = register_annot_v2(annot_move_wkt, img_inst_move, img_inst_static, init_path_imgs, tmp_folder="tmp/", orb=False)
				elif version == "large":
					annot_reg_wkt = register_annot_v2_large(annot_move_wkt, img_inst_move, img_inst_static, init_path_imgs, tmp_folder="tmp/", orb=False)
				elif version == "validation":
					annot_reg_wkt = two_step_validation(annot_move_wkt, img_inst_move, img_inst_static, init_path_imgs, tmp_folder="", orb=True, thresh=thresh)
				else:
					print("Invalid version parameter : " + version +"\n Use \"single\", \"pattern\", \"large\" or \"validation\" instead")
					return

				# metrics
				diag_size = (img_inst_static.width**2 + img_inst_static.height**2)**0.5

				centr_err_base, iou_base = metrics(annot_static_wkt, annot_move_wkt, ref_size=diag_size)

				if annot_reg_wkt is None:
					centr_err_reg, iou_reg = [1,0]
					print("\nUNABLE TO PERFORM REGISTRATION !!!\n")
					annot_grp_fails[-1] +=1
					# full stain name
					stain_dict_fails[stain_move] +=1
					# split stain name
					for stain in stain_move_list:
						stain_dict_split_fails[stain] +=1

					fails+=1
				else:
					centr_err_reg, iou_reg = metrics(annot_static_wkt, annot_reg_wkt, ref_size=diag_size)

				metrics_base[-1][0] += centr_err_base
				metrics_base[-1][1] += iou_base
				metrics_reg[-1][0] += centr_err_reg
				metrics_reg[-1][1] += iou_reg

				mean_centroid_err_base += centr_err_base
				mean_IoU_base += iou_base
				mean_centroid_err_reg += centr_err_reg
				mean_IoU_reg += iou_reg

				annot_grp_n_pairs[-1] += 1
				n_pair += 1

				# full stain name
				add_result_dicts(stain_dict_occ, stain_dict_IoU_base, stain_dict_centroid_err_base,
									stain_dict_IoU_reg, stain_dict_centroid_err_reg,
									stain_move, iou_base, centr_err_base, iou_reg, centr_err_reg)

				# split stain name
				for stain in stain_move_list:
					add_result_dicts(stain_dict_split_occ, stain_dict_split_IoU_base, stain_dict_split_centroid_err_base,
										stain_dict_split_IoU_reg, stain_dict_split_centroid_err_reg,
										stain, iou_base, centr_err_base, iou_reg, centr_err_reg)

				print("########################### PROGRESS ###########################")
				print("annot_grp_n_pairs={}".format(annot_grp_n_pairs))
				print("################################################################")


		# mean on a group
		metrics_base[-1][0] /= annot_grp_n_pairs[-1]
		metrics_base[-1][1] /= annot_grp_n_pairs[-1]
		metrics_reg[-1][0] /= annot_grp_n_pairs[-1]
		metrics_reg[-1][1] /= annot_grp_n_pairs[-1]

		#break


	# mean on all pairs
	mean_centroid_err_base /= n_pair
	mean_IoU_base /= n_pair
	mean_centroid_err_reg /= n_pair
	mean_IoU_reg /= n_pair

	# full stain name
	get_means_dicts(stain_dict_occ, stain_dict_IoU_base, stain_dict_centroid_err_base)
	get_means_dicts(stain_dict_occ, stain_dict_IoU_reg, stain_dict_centroid_err_reg)

	# split stain name
	get_means_dicts(stain_dict_split_occ, stain_dict_split_IoU_base, stain_dict_split_centroid_err_base)
	get_means_dicts(stain_dict_split_occ, stain_dict_split_IoU_reg, stain_dict_split_centroid_err_reg)

	print("\nResults :")
	print("General :")
	print("\tmean_centroid_err_base={}".format(mean_centroid_err_base))
	print("\tmean_IoU_base={}".format(mean_IoU_base))
	print("\tmean_centroid_err_reg={}".format(mean_centroid_err_reg))
	print("\tmean_IoU_reg={}".format(mean_IoU_reg))
	print("\tfails={}".format(fails))
	print()

	print("Per groups :")
	print("annot_grp_ids={}".format(annot_grp_ids))
	print("annot_grp_n_pairs={}".format(annot_grp_n_pairs))
	print("metrics_base={}".format(metrics_base))
	print("metrics_reg={}".format(metrics_reg))
	print()

	print("Per stains (full name) :")
	print_dicts(stain_dict_occ, stain_dict_nbr_ann, stain_dict_imgs, stain_dict_fails,
				stain_dict_IoU_base, stain_dict_centroid_err_base,
				stain_dict_IoU_reg, stain_dict_centroid_err_reg)

	print("Per stains (split name) :")
	print_dicts(stain_dict_split_occ, stain_dict_split_nbr_ann, stain_dict_split_imgs, stain_dict_split_fails,
				stain_dict_split_IoU_base, stain_dict_split_centroid_err_base,
				stain_dict_split_IoU_reg, stain_dict_split_centroid_err_reg)


"""
Compute global statistics on the different annotation group and on the different stains (full and splitted)
(number of annotation, average size of the annotation in the image)

inputs :
	annot_group_coll : 	AnnotationGroupCollection of the different annotation group, can be fetched with :
						AnnotationGroupCollection().fetch_with_filter("project", id_project)
						where <id_project> is the id of the project to test

"""
def annot_groups_stat(annot_group_coll):
	id_list = []
	len_list = []
	grp_name_list = []
	avg_size_list = []

	stain_dict_occ = {}
	stain_dict_nbr_ann = {}
	stain_dict_imgs = {}
	stain_dict_avg_size = {}

	stain_dict_split_occ = {}
	stain_dict_split_nbr_ann = {}
	stain_dict_split_imgs = {}
	stain_dict_split_avg_size = {}

	for annot_group in annot_group_coll:
		annot_link_coll = AnnotationLinkCollection().fetch_with_filter("annotationgroup", annot_group.id)

		# only consider group with at least 2 annotation
		if len(annot_link_coll) <= 1:
			continue

		id_list.append(annot_group.id)
		len_list.append(len(annot_link_coll))

		avg_size = 0
		n_annot = 0
		get_name = True

		for annot_link_move in annot_link_coll:
			annot_move = Annotation(id=annot_link_move.annotationIdent, showWKT=True, showMeta=True, showGIS=True, showLink=True).fetch()
			img_inst_move = ImageInstance(id=annot_move.image).fetch()

			# get image group name (once)
			if get_name:
				name = img_inst_move.instanceFilename
				grp_name_list.append(name.rpartition('-')[0])
				get_name = False

			# get proportion of the annot in the image
			annot_size = annot_move.area/(img_inst_move.width*img_inst_move.height)
			avg_size += annot_size
			n_annot += 1

			# get stain
			stain_move = get_stain(img_inst_move.instanceFilename)
			stain_move_list = split_stain(stain_move)

			# full stain name
			if stain_move in stain_dict_nbr_ann:
				stain_dict_nbr_ann[stain_move] += 1
				stain_dict_imgs[stain_move].add(img_inst_move.originalFilename)
				stain_dict_occ[stain_move] += (len(annot_link_coll)-1)
				stain_dict_avg_size[stain_move] += annot_size
			else:
				stain_dict_nbr_ann[stain_move] = 1
				stain_dict_imgs[stain_move] = {img_inst_move.originalFilename}
				stain_dict_occ[stain_move] = (len(annot_link_coll)-1)
				stain_dict_avg_size[stain_move] = annot_size

			# stains split list
			for stain in stain_move_list:
				if stain in stain_dict_split_nbr_ann:
					stain_dict_split_nbr_ann[stain] += 1
					stain_dict_split_imgs[stain].add(img_inst_move.originalFilename)
					stain_dict_split_occ[stain] += (len(annot_link_coll)-1)
					stain_dict_split_avg_size[stain] += annot_size
				else:
					stain_dict_split_nbr_ann[stain] = 1
					stain_dict_split_imgs[stain] = {img_inst_move.originalFilename}
					stain_dict_split_occ[stain] = (len(annot_link_coll)-1)
					stain_dict_split_avg_size[stain] = annot_size

		#break

		avg_size /= n_annot
		avg_size_list.append(avg_size*100)

	tot_annot = 0
	print('----------------------------------------------------------------------------------------------')
	print("Per groups :")
	for idx in range(len(id_list)):
		print("Annotation group : {}".format(id_list[idx]))
		print("Length = {}".format(len_list[idx]))
		tot_annot += len_list[idx]
		print("Group name : {}".format(grp_name_list[idx]))
		print("Avg size : {}%\n".format(avg_size_list[idx]))

	print("Total number of usable annotation : {}".format(tot_annot))

	# get means (currently only sum is stored)
	for key in stain_dict_nbr_ann:
		stain_dict_avg_size[key] /= (stain_dict_nbr_ann[key] /100)
	for key in stain_dict_split_nbr_ann:
		stain_dict_split_avg_size[key] /= (stain_dict_split_nbr_ann[key] /100)

	print('----------------------------------------------------------------------------------------------')
	print("Per stains (mix) :")
	print("stain\nn_annot\nn_occ\nn_img\navg_size\n")
	for key in stain_dict_nbr_ann:
		print("{}\n{}\n{}\n{}\n{}\n\n".format(key, stain_dict_nbr_ann[key],
												stain_dict_occ[key], len(stain_dict_imgs[key]),
												stain_dict_avg_size[key]))


	print('----------------------------------------------------------------------------------------------')
	print("Per stains (split) :")
	print("stain\nn_annot\nn_occ\nn_img\navg_size\n")
	for key in stain_dict_split_nbr_ann:
		print("{}\n{}\n{}\n{}\n{}\n\n".format(key, stain_dict_split_nbr_ann[key],
												stain_dict_split_occ[key], len(stain_dict_split_imgs[key]),
												stain_dict_split_avg_size[key]))


"""
Perform a two step registration by feature detection (pattern matching method)
REQUIRE CYTOMINE CONNECTION

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_inst_move : 	ImageInstance of the moving image
	img_inst_static :	ImageInstance of the fixed image
	init_path_img :		path of the folder containing the fetched image of the dataset
	tmp_folder :		folder to place the intermediate fetched image
	orb :				True|False -> use the Orb detector (with Hamming matching) if True (default),
										use SIFT (with L2 matching) otherwise
	plot_fig :			[True|False, True|False, True|False]
						-> plot the resulting registration
							1) first global registration
							2) annotation pattern from moving + surrounding in fixed image
							3) second local registration

outputs :
	the registered Polygon annotation

"""
def register_annot_v2(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, tmp_folder="", orb=True, plot_fig=[False, False, False]):

	#perform annotation on whole image
	annot_cent_moved = register_annot_v1_cyt(annot_move_wkt.centroid, img_inst_move, img_inst_static, init_path_img, orb=orb, plot_fig=plot_fig[0])

	if annot_cent_moved is None:
		return None
	### extract sub-images

	# will extract windows based on the annotation + add "inc_perc"*(width|height) on each border
	inc_ref = 0.3
	# will extract windows centered on the registered centroid and
	inc_lookup = 2
	min_perc_lookup = 0.1

	max_dump_size = 512


	## move_window
	# extract bounds (minx, miny, maxx, maxy)
	(minx_move, miny_move, maxx_move, maxy_move) = annot_move_wkt.bounds
	window_move_width = maxx_move - minx_move
	window_move_height = maxy_move - miny_move

	# download dump of the window (left, bottom, width, height)
	window_move = (max(0, floor(minx_move - (inc_ref/2)*window_move_width)),
				  max(0, floor(miny_move - (inc_ref/2)*window_move_height)))
	window_move += (min(ceil((1+inc_ref)*window_move_width), img_inst_move.width-window_move[0]),
					min(ceil((1+inc_ref)*window_move_height), img_inst_move.height-window_move[1]))



	# need top left corner
	img_inst_move.window(window_move[0],
						 img_inst_move.height - window_move[1] - window_move[3],
						 window_move[2],
						 window_move[3],
						 dest_pattern=tmp_folder+"window_move.png", max_size=max_dump_size)

	window_move_PIL = Image.open(tmp_folder+"window_move.png")


	if plot_fig[1]:
		display(window_move_PIL)

	## moved_window
	# download dump of the window (left, bottom, width, height)
	# at least "inc_lookup" times the size of the previous window
	# and at least "min_perc_lookup" the size of the image
	window_moved = [0,0,0,0]

	# get appropriate size
	window_moved[2:4] = [max(ceil(window_move[2]*inc_lookup), ceil(min_perc_lookup*img_inst_static.width)),
						max(ceil(window_move[3]*inc_lookup), ceil(min_perc_lookup*img_inst_static.height))]


	# find bottom left
	window_moved[:2] = [floor(annot_cent_moved.x - (window_moved[2]/2)),
						floor(annot_cent_moved.y - (window_moved[3]/2))]

	# does not go out of the image
	window_moved = valid_window(window_moved, [img_inst_static.width, img_inst_static.height])

	if window_moved is None:
		return None

	# need top left corner
	img_inst_static.window(window_moved[0],
						 img_inst_static.height - window_moved[1] - window_moved[3],
						 window_moved[2],
						 window_moved[3],
						 dest_pattern=tmp_folder+"window_moved.png", max_size=max_dump_size)

	# return None if window messed up such that no windows has been dump
	try:
		window_moved_PIL = Image.open(tmp_folder+"window_moved.png")

	except FileNotFoundError:
		return None

	if plot_fig[1]:
		display(window_moved_PIL)

	# keep the relative proportion of the images
	keep_prop = False
	if keep_prop:
		shrink_move = max(1, max(window_move[2:])/max_dump_size)
		shrink_moved = max(1, max(window_moved[2:])/max_dump_size)

		if shrink_move > shrink_moved:
			window_move_PIL = window_move_PIL.resize((int(window_move_PIL.width*(shrink_move/shrink_moved)),
														int(window_move_PIL.height*(shrink_move/shrink_moved))),
														Image.ANTIALIAS)

		else:
			window_moved_PIL = window_moved_PIL.resize((int(window_moved_PIL.width*(shrink_moved/shrink_move)),
														int(window_moved_PIL.height*(shrink_moved/shrink_move))),
														Image.ANTIALIAS)


	# second registration
	annot_move_loc = translate(annot_move_wkt, xoff=-window_move[0], yoff=-window_move[1])


	annot_moved_loc = register_annot_v1(annot_move_loc, window_move_PIL, window_moved_PIL,
										[window_move[2], window_move[3]],
										[window_moved[2], window_moved[3]],
										orb=orb, plot_fig=plot_fig[2])

	if annot_moved_loc == None:
		return None

	annot_moved_loc = translate(annot_moved_loc, xoff=window_moved[0], yoff=window_moved[1])

	window_move_PIL.close()
	os.remove(tmp_folder+"window_move.png")
	window_moved_PIL.close()
	os.remove(tmp_folder+"window_moved.png")

	return annot_moved_loc


"""
Perform a two step registration by feature detection (large window method)
REQUIRE CYTOMINE CONNECTION

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_inst_move : 	ImageInstance of the moving image
	img_inst_static :	ImageInstance of the fixed image
	init_path_img :		path of the folder containing the fetched image of the dataset
	tmp_folder :		folder to place the intermediate fetched image
	orb :				True|False -> use the Orb detector (with Hamming matching) if True (default),
										use SIFT (with L2 matching) otherwise
	plot_fig :			[True|False, True|False, True|False]
						-> plot the resulting registration
							1) first global registration
							2) annotation pattern from moving + surrounding in fixed image
							3) second local registration

outputs :
	the registered Polygon annotation

"""
def register_annot_v2_large(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, tmp_folder="", orb=True, plot_fig=[False, False, False]):

	#perform annotation on whole image
	annot_cent_moved = register_annot_v1_cyt(annot_move_wkt.centroid, img_inst_move, img_inst_static, init_path_img, orb=orb, plot_fig=plot_fig[0])

	if annot_cent_moved is None:
		return None
	### extract sub-images

	# will extract windows centered on the registered centroid and
	inc_lookup = 2
	min_perc_lookup = 0.1

	max_dump_size = 512

	## move_window
	# extract bounds (minx, miny, maxx, maxy)
	(minx_move, miny_move, maxx_move, maxy_move) = annot_move_wkt.bounds
	window_width = ceil((1+inc_lookup)*(maxx_move - minx_move))
	window_height = ceil((1+inc_lookup)*(maxy_move - miny_move))

	# check if minimal size reached
	min_width = ceil(min_perc_lookup*img_inst_move.width)
	if min_width > window_width:
		window_width = min_width

	min_height = ceil(min_perc_lookup*img_inst_move.height)
	if min_height > window_height:
		window_height = min_height


	# download dump of the window (left, bottom, width, height)
	window_move = [floor(annot_move_wkt.centroid.x - (window_width/2)),
				  	floor(annot_move_wkt.centroid.y - (window_height/2)),
				  	window_width,
				  	window_height]


	print("Check validity (move)")
	window_move = valid_window(window_move, [img_inst_move.width, img_inst_move.height])


	# need top left corner
	img_inst_move.window(window_move[0],
						 img_inst_move.height - window_move[1] - window_move[3],
						 window_move[2],
						 window_move[3],
						 dest_pattern=tmp_folder+"window_move.png", max_size=max_dump_size)

	window_move_PIL = Image.open(tmp_folder+"window_move.png")


	if plot_fig[1]:
		display(window_move_PIL)

	## moved_window
	# download dump of the window (left, bottom, width, height)
	# at least "inc_lookup" times the size of the previous window
	# and at least "min_perc_lookup" the size of the image
	window_moved = [floor(annot_cent_moved.x - (window_width/2)),
					floor(annot_cent_moved.y - (window_height/2)),
					window_width, window_height]

	window_moved = valid_window(window_moved, [img_inst_static.width, img_inst_static.height])
	if window_moved is None:
		window_move_PIL.close()
		return None


	# need top left corner
	img_inst_static.window(window_moved[0],
						 img_inst_static.height - window_moved[1] - window_moved[3],
						 window_moved[2],
						 window_moved[3],
						 dest_pattern=tmp_folder+"window_moved.png", max_size=max_dump_size)

	# return None if window messed up such that no windows has been dump
	try:
		window_moved_PIL = Image.open(tmp_folder+"window_moved.png")

	except FileNotFoundError:
		window_move_PIL.close()
		return None

	if plot_fig[1]:
		display(window_moved_PIL)

	# keep the relative proportion of the images
	keep_prop = False
	if keep_prop:
		shrink_move = max(1, max(window_move[2:])/max_dump_size)
		shrink_moved = max(1, max(window_moved[2:])/max_dump_size)

		if shrink_move > shrink_moved:
			window_move_PIL = window_move_PIL.resize((int(window_move_PIL.width*(shrink_move/shrink_moved)),
														int(window_move_PIL.height*(shrink_move/shrink_moved))),
														Image.ANTIALIAS)

		else:
			window_moved_PIL = window_moved_PIL.resize((int(window_moved_PIL.width*(shrink_moved/shrink_move)),
														int(window_moved_PIL.height*(shrink_moved/shrink_move))),
														Image.ANTIALIAS)


	# second registration
	annot_move_loc = translate(annot_move_wkt, xoff=-window_move[0], yoff=-window_move[1])


	#print(annot_move_loc)

	annot_moved_loc = register_annot_v1(annot_move_loc, window_move_PIL, window_moved_PIL,
										[window_move[2], window_move[3]],
										[window_moved[2], window_moved[3]],
										orb=orb, plot_fig=plot_fig[2])

	if annot_moved_loc == None:
		window_move_PIL.close()
		return None

	annot_moved_loc = translate(annot_moved_loc, xoff=window_moved[0], yoff=window_moved[1])

	window_move_PIL.close()
	os.remove(tmp_folder+"window_move.png")
	window_moved_PIL.close()
	os.remove(tmp_folder+"window_moved.png")

	return annot_moved_loc

"""
Make sure a window for dump is included in the image

inputs :
	window : 	4-values list describing the window [left, bottom, width, height]
	img_size :	size of the image from which the window have to be dumped

outputs :
	window :	window whose parts out of the image have been removed
				(identity if the window is already in the image)
"""
def valid_window(window, img_size):
	# check if it is fully out of the image
	if (window[0] > img_size[0]) or (window[1] > img_size[1]):
		return None


	# do not go out of border
	# left
	if window[0] < 0:
		# resize width
		window[2] += window[0]
		# set left to 0
		window[0] = 0

	# bottom
	if window[1] < 0:
		# resize width
		window[3] += window[1]
		# set left to 0
		window[1] = 0

	# right
	if (window[0] + window[2]) > img_size[0]:
		# resize width
		window[2] = img_size[0] - window[0]

	# top
	if (window[1] + window[3]) > img_size[1]:
		# resize width
		window[3] = img_size[1] - window[1]

	return window


"""
Perform a one step and a two steps registration by feature detection (large window method for the two step),
accept the two steps if its IoU with the one step is large enough
REQUIRE CYTOMINE CONNECTION

NB: this implementation is a preliminary version performing the first step twice by using the functions available above,
	it could be speed up by re-implementing the registrations such that the first step is performed only once

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_inst_move : 	ImageInstance of the moving image
	img_inst_static :	ImageInstance of the fixed image
	init_path_img :		path of the folder containing the fetched image of the dataset
	tmp_folder :		folder to place the intermediate fetched image
	orb :				True|False -> use the Orb detector (with Hamming matching) if True (default),
										use SIFT (with L2 matching) otherwise
	thresh :			[0,1] -> determine how much the one step and two steps should overlap (in IoU) to accept the two steps

outputs :
	the registered Polygon annotation

"""
def two_step_validation(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, tmp_folder="", orb=True, thresh=0.5):


	annot_reg_V1 = register_annot_v1_cyt(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, orb=orb)
	# annot_reg_wkt = register_annot_v2(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, tmp_folder="tmp/", orb=orb)
	annot_reg_wkt_V2_large = register_annot_v2_large(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, tmp_folder="tmp/", orb=orb)

	# return V1 if V2 fail
	if annot_reg_V1 == None:
		return None
	elif annot_reg_wkt_V2_large == None:
		return annot_reg_V1

	iou = annot_reg_V1.intersection(annot_reg_wkt_V2_large).area/ annot_reg_V1.union(annot_reg_wkt_V2_large).area
	print(iou)

	return annot_reg_V1 if iou < thresh else annot_reg_wkt_V2_large