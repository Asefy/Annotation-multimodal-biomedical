# Annotation-multimodal-biomedical

This github contains the codes as well as the conda environments used for the master thesis : "AI-assisted annotation of large andmultimodal imaging datasets". This project was done in collaboration with **Cytomine** (https://cytomine.be/) whose Python API can be found at https://github.com/Cytomine-ULiege/Cytomine-python-client/releases.

* The folder **"MI (SITK)"** contains the codes to perform the pixel-based registrations. The codes run using the SITK library on Python 3.7 and at least Cytomine 2.8.0.

* The folder **"ORB-SIFT (OpenCV)"** contains the codes to perform the feature-based registrations as well as the codes to create the dataset used in the **"U-net (Pytorch)"**. The codes run using the OpenCV library on Python 3.8 and at least Cytomine 2.8.1.

* The folder **"U-net (Pytorch)"** contains the codes used for the deep learning segmentation. The codes run using the PyTorch library on Python 3.9.


## MI and ORB-SIFT
To use the codes in these two folders, the images should first be fetched. This can be done through the jupyter notebook **fetch_imgs.ipynb** using Cytomine 2.8.0 at least (the environments in MI and ORB-SIFT should work). The images will be fetched in the **"ORB-SIFT (OpenCV)/"** folder but the folder created, "fetched_imgs/" can be transfered to the **"MI (SITK)/"** folder to run those codes.

Jupyter notebooks **register_all.ipynb** are available in both folders to run the registration on all pairs.

## U-net
To use the U-net segmentatio. The specific dataset should first be created in "ORB-SIFT (OpenCV)/" folder. The dataset will then be stored in the "dataset_custom/" subfolder that should be transfered to the "U-net (Pytorch)/" folder.
