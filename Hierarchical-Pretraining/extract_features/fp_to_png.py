import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
# from models.resnet_custom import resnet50_baseline
import argparse
# from utils.utils import print_network, collate_features
from utils import save_batch_images, collate_features
from PIL import Image
import h5py
import openslide
import sys
sys.path.append('/remote-home/share/Carboxy/Code/HIPT/')
from HIPT_4K.hipt_4k import HIPT_256

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save patched images (.png file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, ignore_augmentation=True)
	x, y = dataset[0]
	kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)

			images = batch.permute(0, 2, 3, 1) # [N, 3, 4096, 4096] -> [N, 4096, 4096, 3]
			images = images.numpy()
			images = (images * 255).astype(np.uint8)

			asset_dict = {'images': images, 'coords': coords}
			save_batch_images(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
# parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
# parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--image_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
# parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()
# python Hierarchical-Pretraining/extract_features/extract_features_fp_4k.py --data_h5_dir /remote-home/share/DATA/tcga-4096-processed-demo/ --image_dir /remote-home/share/DATA/tcga-4096-processed-demo/images/ --target_patch_size 4096

if __name__ == '__main__':

	print('initializing dataset')
	# csv_path = args.csv_path
	# if csv_path is None:
	# 	raise NotImplementedError
	csv_path = os.path.join(args.data_h5_dir, 'process_list_autogen.csv')
	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.image_dir, exist_ok=True)
	# os.makedirs(os.path.join(args.feat_dir, 'image'), exist_ok=True)
	# os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	# dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))


	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0].split('/')[-1]

		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		# slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		slide_file_path = bags_dataset[bag_candidate_idx]
		print('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
		print(slide_id)

		# if not args.no_auto_skip and slide_id+'.pt' in dest_files:
		# 	print('skipped {}'.format(slide_id))
		# 	continue 

		output_path = os.path.join(args.image_dir, slide_id)
		os.makedirs(output_path, exist_ok=True)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)

		# Determine whether downsampling is required
		# reference from https://github.com/binli123/dsmil-wsi/issues/32 and https://github.com/mahmoodlab/CLAM/issues/163
		MAG_BASE = bags_dataset.df['MAG_BASE'][bag_candidate_idx]
		MPP_X = bags_dataset.df['MPP_X'][bag_candidate_idx]
		level_downsamples = eval(bags_dataset.df['level_downsamples'][bag_candidate_idx])
		if (MPP_X > 0.4) and (MPP_X < 0.6):
			# if MAG_BASE == 20:
			# 	custom_downsample_this = 1
			# 	target_patch_size_this = args.target_patch_size
			# else:
			# 	print('skipped {}'.format(slide_id), ', because of invalid MAG_BASE.')
			# 	continue
			custom_downsample_this = 1
			target_patch_size_this = args.target_patch_size
		elif (MPP_X > 0.2) and (MPP_X < 0.3):
			custom_downsample_this = 2
			target_patch_size_this = args.target_patch_size	
		else:
			print('skipped {}'.format(slide_id), ', because of invalid MPP_X.')
			continue

		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=custom_downsample_this, target_patch_size=target_patch_size_this,
		pretrained=True)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))




