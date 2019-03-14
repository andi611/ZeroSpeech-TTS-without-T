# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ loss.py ]
#   Synopsis     [ Loss for the Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn
from model.tacotron.config import config


#################
# TACOTRON LOSS #
#################
class TacotronLoss(nn.Module):
	def __init__(self):
		super(TacotronLoss, self).__init__()
		
		self.sample_rate = config.sample_rate
		self.linear_dim = config.num_freq
		self.prior_freq = config.prior_freq
		self.prior_weight = config.prior_weight
		self.gate_coefficient = config.gate_coefficient

		self.criterion = nn.MSELoss()
		self.criterion_gate = nn.BCEWithLogitsLoss()

	def forward(self, model_output, targets):
		mel_outputs, mel = model_output[0], targets[0]
		linear_outputs, linear = model_output[1], targets[1]
		gate_outputs, gate = model_output[2], targets[2]

		mel.requires_grad = False
		linear.requires_grad = False
		gate.requires_grad = False

		mel_loss = self.criterion(mel_outputs, mel)
		n_priority_freq = int(self.prior_freq / (self.sample_rate * 0.5) * self.linear_dim)
		linear_loss = (1 - self.prior_weight) * self.criterion(linear_outputs, linear) + self.prior_weight * self.criterion(linear_outputs[:, :, :n_priority_freq], linear[:, :, :n_priority_freq])
		gate_loss = self.gate_coefficient * self.criterion_gate(gate_outputs, gate)
		
		loss = mel_loss + linear_loss + gate_loss
		losses = [loss, mel_loss, linear_loss, gate_loss]
		return losses


#########################
# GET MASK FROM LENGTHS #
#########################
"""
	Get mask tensor from list of length

	Args:
		memory: (batch, max_time, dim)
		memory_lengths: array like
"""
def get_rnn_mask_from_lengths(memory, memory_lengths):
	mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
	for idx, l in enumerate(memory_lengths):
		mask[idx][:l] = 1
	return ~mask

