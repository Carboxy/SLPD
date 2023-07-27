import pickle
import h5py
import torch
import numpy as np
from PIL import Image
import os

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def save_batch_images(output_path, asset_dict, attr_dict= None, mode='a'):
    images = asset_dict['images'] # [N, 4096, 4096, 3], numpy.array, np.uint8
    coords = asset_dict['coords'] # [N, 2], numpy.array
    for image, coord in zip(images, coords):
        coord_x = coord[0]
        coord_y = coord[1]
        silde_id = output_path.split('/')[-1]
        image_name = silde_id+'_'+str(coord_x)+'_'+str(coord_y)+'.png'
        image_pil = Image.fromarray(image, mode='RGB')
        image_pil.save(os.path.join(output_path, image_name))

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]