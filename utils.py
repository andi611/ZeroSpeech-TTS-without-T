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
import torch
import numpy as np
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


def cal_acc(logits, y_true, shift=False):
	_, ind = torch.max(logits, dim=1)
	if shift:
		acc = torch.sum((ind == y_true - 100).type(torch.FloatTensor)) / y_true.size(0)
	else:
		acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
	return acc


class Logger(object):
	def __init__(self, log_dir='./log'):
		self.writer = SummaryWriter(log_dir)

	def scalar_summary(self, tag, value, step):
		self.writer.add_scalar(tag, value, step)

