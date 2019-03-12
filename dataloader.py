# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataloader.py ]
#   Synopsis     [ Dataset wrapper and data loader ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import json
import h5py
import torch
import numpy as np
from torch.utils import data
from collections import namedtuple


class DataLoader(object):
	def __init__(self, dataset, batch_size=16):
		self.dataset = dataset
		self.n_elements = len(self.dataset[0])
		self.batch_size = batch_size
		self.index = 0

	def all(self, size=1000):
		samples = [self.dataset[self.index + i] for i in range(size)]
		batch = [[s for s in sample] for sample in zip(*samples)]
		batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]

		if self.index + 2 * self.batch_size >= len(self.dataset):
			self.index = 0
		else:
			self.index += self.batch_size
		return tuple(batch_tensor)

	def __iter__(self):
		return self

	def __next__(self):
		samples = [self.dataset[self.index + i] for i in range(self.batch_size)]
		batch = [[s for s in sample] for sample in zip(*samples)]
		batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]

		if self.index + 2 * self.batch_size >= len(self.dataset):
			self.index = 0
		else:
			self.index += self.batch_size
		return tuple(batch_tensor)


class Dataset(data.Dataset):
	def __init__(self, h5_path, index_path, dset='train', seg_len=64, load_mel=False):
		self.dataset = h5py.File(h5_path, 'r')
		with open(index_path) as f_index:
			self.indexes = json.load(f_index)
		self.indexer = namedtuple('index', ['speaker', 'i', 't'])
		self.seg_len = seg_len
		self.dset = dset

	def __getitem__(self, i):
		index = self.indexes[i]
		index = self.indexer(**index)
		speaker = index.speaker
		i, t = index.i, index.t
		seg_len = self.seg_len
		if load_mel:
			data = [speaker, self.dataset[f'{self.dset}/{i}/lin'][t:t+seg_len], self.dataset[f'{self.dset}/{i}/mel'][t:t+seg_len]]
		else:
			data = [speaker, self.dataset[f'{self.dset}/{i}/lin'][t:t+seg_len]]
		return tuple(data)

	def __len__(self):
		return len(self.indexes)

