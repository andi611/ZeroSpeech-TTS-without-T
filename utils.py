# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils.py ]
#   Synopsis     [ utility functions: training utils / Hps / Logger ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import json
import h5py
import torch
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
from torch.autograd import Variable


def cc(net):
	if torch.cuda.is_available():
		return net.cuda()
	else:
		return net

def gen_noise(x_dim, y_dim):
	x = torch.randn(x_dim, 1) 
	y = torch.randn(1, y_dim)
	return x @ y

def cal_mean_grad(net):
	grad = Variable(torch.FloatTensor([0])).cuda()
	for i, p in enumerate(net.parameters()):
		grad += torch.mean(p.grad)
	return grad.data[0] / (i + 1)


def multiply_grad(nets, c):
	for net in nets:
		for p in net.parameters():
			p.grad *= c

def to_var(x, requires_grad=True):
	x = Variable(x, requires_grad=requires_grad)
	return x.cuda() if torch.cuda.is_available() else x


def reset_grad(net_list):
	for net in net_list:
		net.zero_grad()


def grad_clip(net_list, max_grad_norm):
	for net in net_list:
		torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


def calculate_gradients_penalty(netD, real_data, fake_data):
	alpha = torch.rand(real_data.size(0))
	alpha = alpha.view(real_data.size(0), 1, 1)
	alpha = alpha.cuda() if torch.cuda.is_available() else alpha
	alpha = Variable(alpha)
	interpolates = alpha * real_data + (1 - alpha) * fake_data

	disc_interpolates = netD(interpolates)

	use_cuda = torch.cuda.is_available()
	grad_outputs = torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(disc_interpolates.size())

	gradients = torch.autograd.grad(
		outputs=disc_interpolates,
		inputs=interpolates,
		grad_outputs=grad_outputs,
		create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients_penalty = (1. - torch.sqrt(1e-12 + torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1))) ** 2
	gradients_penalty = torch.mean(gradients_penalty)
	return gradients_penalty


def cal_acc(logits, y_true):
	_, ind = torch.max(logits, dim=1)
	acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
	return acc


class Hps(object):
	def __init__(self, path=None):
		self.hps = namedtuple('hps', [
			'lr',
			'alpha_dis',
			'alpha_enc',
			'beta_dis', 
			'beta_gen', 
			'beta_clf',
			'lambda_',
			'ns', 
			'enc_dp', 
			'dis_dp', 
			'max_grad_norm',
			'max_step',
			'seg_len',
			'n_samples',
			'emb_size',
			'n_speakers',
			'n_target_speakers',
			'n_latent_steps',
			'n_patch_steps', 
			'batch_size',
			'lat_sched_iters',
			'enc_pretrain_iters',
			'dis_pretrain_iters',
			'patch_iters', 
			'iters',
			'max_to_keep',
			]
		)
		if not path is None:
			self.load(path)
		else:
			print('Using default parameters since no .json file is provided.')
			default = \
				[1e-4, 1, 1e-4, 0, 0, 0, 10, 0.01, 0.5, 0.1, 5, 5, 128, 400000, 128, 102, 2, 5, 0, 32, 50000, 5000, 5000, 30000, 60000, 10]
			self._hps = self.hps._make(default)

	def get_tuple(self):
		return self._hps

	def load(self, path):
		with open(path, 'r') as f_json:
			hps_dict = json.load(f_json)
		self._hps = self.hps(**hps_dict)

	def dump(self, path):
		with open(path, 'w') as f_json:
			json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))


class Logger(object):
	def __init__(self, log_dir='./log'):
		self.writer = SummaryWriter(log_dir)

	def scalar_summary(self, tag, value, step):
		self.writer.add_scalar(tag, value, step)

