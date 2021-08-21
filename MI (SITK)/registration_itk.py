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
import SimpleITK as sitk

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
complete/add the entries for the different metrics for the key "name"

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
def print_dicts(dict_occ, dict_n_img, dict_imgs, dict_fails, dict_IoU_base, dict_err_base, dict_IoU_reg, dict_err_reg):
	print("stain\nn_img\nn_occ\nimages\nIoU_base\ncent_err_base\nIoU_reg\ncent_err_reg\nfails\n")
	for key in dict_occ:
		print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(key, dict_occ[key], dict_n_img[key], dict_imgs[key],
													dict_IoU_base[key], dict_err_base[key],
													dict_IoU_reg[key], dict_err_reg[key], dict_fails[key]))

"""
Perform a single step registration using gradient descent and mutual information,
wrapper for the register_annot_v1() using Cytomine ImageInstance as input instead

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_inst_move : 	ImageInstance of the moving image
	img_inst_static :	ImageInstance of the fixed image
	init_path_img :		path of the folder containing the fetched image of the dataset
	full_aff :			True|False -> try to find a full affine transform if True,
										find an affine without shearing otherwise (default)
	plot_fig :			True|False -> plot the resulting registration

outputs :
	new_annot :		the registered Polygon annotation

"""
def register_annot_v1_cyt(annot_move_wkt, img_inst_move, img_inst_static, init_path_img, full_aff=False, plot_fig=False):
	img_move_PIL = Image.open(init_path_img+img_inst_move.originalFilename+".jpg")

	img_static_PIL = Image.open(init_path_img+img_inst_static.originalFilename+".jpg")


	new_annot =  register_annot_SITK(annot_move_wkt, img_move_PIL, img_static_PIL,
										[img_inst_move.width, img_inst_move.height],
										[img_inst_static.width, img_inst_static.height],
										init_path_img, full_aff=full_aff, plot_fig=plot_fig)

	img_move_PIL.close()
	img_static_PIL.close()

	return new_annot


"""
Perform registration for all pairs and summarize the results for the IoU and RCM metrics
REQUIRE CYTOMINE CONNECTION

inputs :
	annot_group_coll : 	AnnotationGroupCollection of the different annotation group, can be fetched with :
						AnnotationGroupCollection().fetch_with_filter("project", id_project)
						where <id_project> is the id of the project to test
	init_path_imgs :	path of the folder containing the fetched image of the dataset
	full_aff :			True|False -> try to find a full affine transform if True (ONLY AVAILABLE FOR V1),
										find an affine without shearing otherwise (default)

"""
def metrics_reg_all(annot_group_coll, init_path_imgs="dl-file/526102245/reduced_img/", full_aff=False):

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

	i=0
	j=0

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

			# extract stain
			stain_move = get_stain(img_inst_move.instanceFilename)

			# add stain entry
			if stain_move in stain_dict_nbr_ann:
				stain_dict_nbr_ann[stain_move] += 1
				stain_dict_imgs[stain_move].add(img_inst_move.originalFilename)
			else:
				stain_dict_nbr_ann[stain_move] = 1
				stain_dict_imgs[stain_move] = {img_inst_move.originalFilename}
				stain_dict_fails[stain_move] = 0

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
				annot_reg_wkt = register_annot_v1_cyt(annot_move_wkt, img_inst_move, img_inst_static, init_path_imgs, full_aff=full_aff)

				# metrics
				diag_size = (img_inst_static.width**2 + img_inst_static.height**2)**0.5

				centr_err_base, iou_base = metrics(annot_static_wkt, annot_move_wkt, ref_size=diag_size)

				if annot_reg_wkt is None:
					centr_err_reg, iou_reg = [1,0]
					print("\nUNABLE TO PERFORM REGISTRATION !!!\n")
					annot_grp_fails[-1] +=1
					stain_dict_fails[stain_move] +=1
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

				add_result_dicts(stain_dict_occ, stain_dict_IoU_base, stain_dict_centroid_err_base,
									stain_dict_IoU_reg, stain_dict_centroid_err_reg,
									stain_move, iou_base, centr_err_base, iou_reg, centr_err_reg)

				print("annot_grp_n_pairs={}".format(annot_grp_n_pairs))


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

	get_means_dicts(stain_dict_occ, stain_dict_IoU_base, stain_dict_centroid_err_base)
	get_means_dicts(stain_dict_occ, stain_dict_IoU_reg, stain_dict_centroid_err_reg)

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

	print("Per stains :")
	print_dicts(stain_dict_occ, stain_dict_nbr_ann, stain_dict_imgs, stain_dict_fails,
				stain_dict_IoU_base, stain_dict_centroid_err_base,
				stain_dict_IoU_reg, stain_dict_centroid_err_reg)


"""
Perform a single step registration using gradient descent and mutual information

inputs :
	annot_move_wkt :	Polygon of the annotation in the moving image
	img_move_PIL : 		PIL image of the moving image
	img_static_PIL :	PIL image of the fixed image
	orig_move_size :	original size of the moving image (before fetching)
	orig_static_size :	original size of the fixed image (before fetching)
	init_path_img :		path of the folder containing the fetched image of the dataset
	full_aff :			True|False -> try to find a full affine transform if True,
										find an affine without shearing otherwise (default)
	plot_fig :			True|False -> plot the resulting registration

outputs :
	new_annot :		the registered Polygon annotation

"""
def register_annot_SITK(annot_move_wkt, img_move_PIL, img_static_PIL, orig_move_size, orig_static_size, init_path_img, full_aff=False, plot_fig=False):
	#display(img_move_PIL)
	#display(img_static_PIL)

	## Get sitk greyscale image
	img_move_sitk = sitk.Cast(sitk.GetImageFromArray(np.asarray(ImageOps.grayscale(img_move_PIL))), sitk.sitkFloat32)
	img_static_sitk = sitk.Cast(sitk.GetImageFromArray(np.asarray(ImageOps.grayscale(img_static_PIL))), sitk.sitkFloat32)

	# define de degree of freedom of the transform
	# "A similarity 2D transform with rotation in radians and isotropic scaling around a fixed center with translation"
	if not full_aff:
		trans = sitk.Similarity2DTransform()

	else:
		trans = sitk.AffineTransform(2)

	# Align center of both image and set center of rotation at this center
	initial_transform = sitk.CenteredTransformInitializer(img_static_sitk,
													  img_move_sitk,
													  trans,
													  sitk.CenteredTransformInitializerFilter.GEOMETRY)

	# Create registration object
	registration_method = sitk.ImageRegistrationMethod()
	registration_method.SetInitialTransform(initial_transform, inPlace=False)

	# Choose similarity metric settings
	method = 0
	if method == 0:
		registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
		rdm_sample_perc = 0.5

	elif method == 1:
		registration_method.SetMetricAsJointHistogramMutualInformation()
		rdm_sample_perc = 0.5


	elif method == 2:
		registration_method.SetMetricAsMeanSquares()
		rdm_sample_perc = 0.5

	else:
		registration_method.SetMetricAsCorrelation()
		rdm_sample_perc = 0.5

	rdm_sampling = False
	if rdm_sampling:
		registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
		registration_method.SetMetricSamplingPercentage(rdm_sample_perc)
	else:
		registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)

	# Choose interpolator (to estimate non-grid points)
	registration_method.SetInterpolator(sitk.sitkLinear)

	# Choose optimizer settings

	registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=200, convergenceMinimumValue=1e-9, convergenceWindowSize=20)
	#registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1, numberOfIterations=1000, convergenceMinimumValue=1e-6, convergenceWindowSize=20)
	#registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.2, numberOfIterations=1000, minStep=1e-6)

	# Perform in multi resolution
	registration_method.SetOptimizerScalesFromPhysicalShift()
	registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
	registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
	registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

	#registration_method.SetNumberOfThreads(1)

	# Perform optimization
	try:
		final_transform = registration_method.Execute(img_static_sitk, img_move_sitk)

	except RuntimeError :
		return None


	### plot transformation
	if plot_fig:
		print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
		show_sitk(img_static_sitk, 'Fixed')
		show_sitk(img_move_sitk, 'To Move')
		resampled = resample_sitk(img_move_sitk, final_transform, reference_image=img_static_sitk)
		show_sitk(resampled, 'Moved')

	# Get transformation matrix for annotation
	inv_trans = final_transform.GetInverse()
	param = inv_trans.GetParameters()
	inv_c = inv_trans.GetFixedParameters()
	affine_t = [param[0]*np.cos(param[1]),
			-param[0]*np.sin(param[1]),
			param[0]*np.sin(param[1]),
			param[0]*np.cos(param[1]),
			param[2] - param[0]*np.cos(param[1])*inv_c[0] + param[0]*np.sin(param[1])*inv_c[1] + inv_c[0],
			param[3] - param[0]*np.sin(param[1])*inv_c[0] - param[0]*np.cos(param[1])*inv_c[1] + inv_c[1]]

	# transfom annotation
	# scale down to image PIL dump
	annot_move_wkt = rescale_annot(annot_move_wkt, orig_move_size, [img_move_PIL.width, img_move_PIL.height])
	# change to sitk coordinate
	annot_move_wkt = affine_transform(annot_move_wkt, [1, 0, 0, -1, 0, img_move_PIL.height])
	# apply affine transformation
	annot_move_wkt = affine_transform(annot_move_wkt, affine_t)
	# go back to cartesian coordinate
	annot_move_wkt = affine_transform(annot_move_wkt, [1, 0, 0, -1, 0, img_static_PIL.height])
	# rescale up to original size
	annot_move_wkt = rescale_annot(annot_move_wkt, [img_static_PIL.width, img_static_PIL.height], orig_static_size)

	return annot_move_wkt


"""
http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html

Function to show a SITK image

inputs :
	img :	SITK image
"""
def show_sitk(img, title=None, margin=0.05, dpi=80):
	nda = sitk.GetArrayViewFromImage(img)
	spacing = img.GetSpacing()

	ysize = nda.shape[0]
	xsize = nda.shape[1]

	figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

	fig = plt.figure(title, figsize=figsize, dpi=dpi)
	ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

	extent = (0, xsize*spacing[1], 0, ysize*spacing[0])

	t = ax.imshow(nda,
			extent=extent,
			interpolation='hamming',
			cmap='gray',
			origin='lower')

	if(title):
		plt.title(title)


"""
# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html

Function to apply an SITK transform to an SITK image

inputs :
	image :		SITK image
	transform :	SITK transform
"""
def resample_sitk(image, transform, reference_image=None):
	# Output image Origin, Spacing, Size, Direction are taken from the reference
	# image in this call to Resample
	if reference_image is None:
		reference_image = image
	interpolator = sitk.sitkCosineWindowedSinc
	default_value = 100.0
	return sitk.Resample(image, reference_image, transform,
						 interpolator, default_value)