'''
Data feeding class. It generates a list of data samples, each of which is a python list of
tuples. Every tuple consists of an image path and a beam index. Since this class is used in
the baseline solution, it only outputs sequences of beam indices, and it ignores the images.
-------------------------------
Author: Muhammad Alrabeiah
Jan. 2020
'''

import os
import numpy as np
import pandas as pd
import torch
import random
from skimage import io
from torch.utils.data import Dataset


############### Create data sample list #################
def create_samples(root, shuffle=False, nat_sort=False):
	f = pd.read_csv(root)
	data_samples = []
	for idx, row in f.iterrows():
		beams = row.values[0:13].astype(np.float32)
		img_paths = row.values[13:]
		sample = list( zip(img_paths,beams) )
		data_samples.append(sample)

	if shuffle:
		random.shuffle(data_samples)
	print('list is ready')
	return data_samples


#########################################################

class DataFeed(Dataset):
	"""
	A class fetching a PyTorch tensor of beam indices.
	"""

	def __init__(self, root_dir,
				n,
				img_dim,
				transform=None,
				init_shuflle=True):

		self.root = root_dir
		self.samples = create_samples(self.root, shuffle=init_shuflle)
		self.transform = transform
		self.seq_len = n
		self.img_dim = img_dim

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx] # Read one data sample
		assert len(sample) >= self.seq_len, 'Unsupported sequence length'
		sample = sample[:self.seq_len] # Read a sequence of tuples from a sample
		beams = torch.zeros((self.seq_len,))
		for i,s in enumerate( sample ):
			x = s[1] # Read only beams
			beams[i] = torch.tensor(x, requires_grad=False)

		return beams