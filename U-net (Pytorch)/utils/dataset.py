import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Dataset structure for the project used "KI-gastrointestinal"
# the repartition between train|evaluation|test sets
# should be changed for another project (or if the dataset project grows)
class BasicDataset(Dataset):
	def __init__(self, img_dir, img_scale=1., my_set="train", data_aug=False):
		self.img_dir = img_dir
		self.img_scale = img_scale
		self.trans = transforms.RandomAffine(45, translate=(0.3,0.3), scale=(0.8,1.2), shear=(-10,10,-10,10))
		self.data_aug = data_aug
		self.short_filenames = []

		if my_set == "train":
			group_list = ("PKR-1-", "PKR-2-", "PKR-7-", "PKR-10-")
		elif my_set == "eval":
			group_list = ("PKR-3-", "PKR-4-")
		elif my_set == "test":
			group_list = ("PKR-5-", "PKR-6-", "PKR-8-", "PKR-9-")
		else:
			print("Invalid set type {}. (train, eval, test) expected.".format(my_set))
			raise Exception()

		# last image built in "build_dataset.py"
		end_file = "+window_static_mask_dataset.png"
		for file in os.listdir(img_dir):
			if file.endswith(end_file) and file.startswith(group_list):
				short_name = file[:-len(end_file)]

				# verify that all images are present with apporiate size
				try:
					img1 = Image.open(img_dir+short_name+"+window_moved_dataset.png")
					img2 = Image.open(img_dir+short_name+"+window_moved_mask_dataset.png")
					img3 = Image.open(img_dir+short_name+"+window_static_dataset.png")
					img4 = Image.open(img_dir+short_name+"+window_static_mask_dataset.png")
				except FileNotFoundError:
					continue

				if (((img1.width != 512) or (img1.height != 512)) or
					((img2.width != 512) or (img2.height != 512)) or
					((img3.width != 512) or (img3.height != 512)) or
					((img4.width != 512) or (img4.height != 512))):
					continue

				self.short_filenames.append(short_name)


	def __len__(self):
		return len(self.short_filenames)


	def rescale(self, pil_img):
		w, h = pil_img.size
		newW, newH = int(self.img_scale * w), int(self.img_scale * h)
		assert newW > 0 and newH > 0, 'Scale is too small'
		pil_img = pil_img.resize((newW, newH))

		return pil_img

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.short_filenames[idx])

		# load the 4 images
		image_moved = Image.open(img_path+"+window_moved_dataset.png")
		image_moved_mask = Image.open(img_path+"+window_moved_mask_dataset.png")
		image_fixed = Image.open(img_path+"+window_static_dataset.png")
		image_fixed_mask = Image.open(img_path+"+window_static_mask_dataset.png")

		# rescale if needed
		if self.img_scale != 1:
			image_moved = self.rescale(image_moved)
			image_moved_mask = self.rescale(image_moved_mask)
			image_fixed = self.rescale(image_fixed)
			image_fixed_mask = self.rescale(image_fixed_mask)

		# convert to grayscale + tensor
		image_moved_gray_tens = transforms.ToTensor()(transforms.functional.to_grayscale(image_moved))
		image_moved_mask_tens = transforms.ToTensor()(image_moved_mask)
		image_fixed_gray_tens = transforms.ToTensor()(transforms.functional.to_grayscale(image_fixed))
		image_fixed_mask_tens = transforms.ToTensor()(image_fixed_mask)

		# directly return images if no transform
		if not self.data_aug:
			image_input = torch.cat([image_moved_gray_tens, image_moved_mask_tens, image_fixed_gray_tens])

			return image_input, image_fixed_mask_tens

		# concatenate for transformation
		image_moved = torch.cat([image_moved_gray_tens, image_moved_mask_tens])
		image_fixed = torch.cat([image_fixed_gray_tens, image_fixed_mask_tens])

		# transform images
		image_moved = self.trans(image_moved)
		image_fixed = self.trans(image_fixed)

		# concatenate the 4 images as needed (3 first for input and last mask for output)
		image_fixed_gray_tens, image_fixed_mask_tens = torch.split(image_fixed, [1, 1])
		image_input = torch.cat([image_moved, image_fixed_gray_tens])

		return image_input, image_fixed_mask_tens
