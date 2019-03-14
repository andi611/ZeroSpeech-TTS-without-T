# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ attention.py ]
#   Synopsis     [ Sequence to sequence attention module for Tacotron ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


######################
# BAHDANAU ATTENTION #
######################
class BahdanauAttention(nn.Module):
	def __init__(self, dim):
		super(BahdanauAttention, self).__init__()
		self.query_layer = nn.Linear(dim, dim, bias=False)
		self.tanh = nn.Tanh()
		self.v = nn.Linear(dim, 1, bias=False)

	"""
		Args:
			query: (batch, 1, dim) or (batch, dim)
			processed_memory: (batch, max_time, dim)
	"""
	def forward(self, query, processed_memory):
		if query.dim() == 2:
			query = query.unsqueeze(1) # insert time-axis for broadcasting

		processed_query = self.query_layer(query) # (batch, 1, dim)
		alignment = self.v(self.tanh(processed_query + processed_memory)) # (batch, max_time, 1)
		alignment = alignment.squeeze(-1) # (batch, max_time)
		
		return alignment


################
# LINEAR LAYER #
################
class LinearNorm(nn.Module):
	
	def __init__(self, 
				 in_dim, 
				 out_dim, 
				 bias=True, 
				 w_init_gain='linear'):

		super(LinearNorm, self).__init__()
		self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

		nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

	def forward(self, x):
		return self.linear_layer(x)


#####################
# CONVOLUTION LAYER #
#####################
class ConvNorm(nn.Module):
	
	def __init__(self, 
				 in_channels, 
				 out_channels, 
				 kernel_size=1, 
				 stride=1,
				 padding=None, 
				 dilation=1, 
				 bias=True, 
				 w_init_gain='linear'):

		super(ConvNorm, self).__init__()
		if padding is None:
			assert(kernel_size % 2 == 1)
			padding = int(dilation * (kernel_size - 1) / 2)

		self.conv = nn.Conv1d(in_channels, out_channels,
							  kernel_size=kernel_size, stride=stride,
							  padding=padding, dilation=dilation,
							  bias=bias)

		nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

	def forward(self, signal):
		conv_signal = self.conv(signal)
		return conv_signal


##################
# LOCATION LAYER #
##################
class LocationLayer(nn.Module):
	
	def __init__(self, 
				 attention_n_filters, 
				 attention_kernel_size,
				 attention_dim):

		super(LocationLayer, self).__init__()
		padding = int((attention_kernel_size - 1) / 2)
		self.location_conv = ConvNorm(2, attention_n_filters,
									  kernel_size=attention_kernel_size,
									  padding=padding, 
									  bias=False, 
									  stride=1,
									  dilation=1)
		self.location_dense = LinearNorm(attention_n_filters, 
										 attention_dim,
										 bias=False,
										 w_init_gain='tanh')

	def forward(self, attention_weights_cat):
		processed_attention = self.location_conv(attention_weights_cat)
		processed_attention = processed_attention.transpose(1, 2)
		processed_attention = self.location_dense(processed_attention)
		return processed_attention


################################
# LOCATION SENSITIVE ATTENTION #
################################
class LocationSensitiveAttention(nn.Module):
	
	def __init__(self, 
				 dim, 
				 attention_location_n_filters=32, 
				 attention_location_kernel_size=31):

		super(LocationSensitiveAttention, self).__init__()
		self.query_layer = LinearNorm(dim, dim, bias=False, w_init_gain='tanh')
		self.location_layer = LocationLayer(attention_location_n_filters,
											attention_location_kernel_size,
											dim)
		self.tanh = nn.Tanh()
		self.v = LinearNorm(dim, 1, bias=False)

	"""
		Args:
			query: (batch, 1, dim) or (batch, dim)
			processed_memory: (batch, max_time, dim)
			attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
	"""
	def forward(self, 
				query, 
				processed_memory,
				attention_weights_cat):

		if query.dim() == 2:
			query = query.unsqueeze(1) # insert time-axis for broadcasting

		processed_query = self.query_layer(query) # (batch, 1, dim)
		processed_attention_weights = self.location_layer(attention_weights_cat)
		alignment = self.v(self.tanh(processed_query + processed_attention_weights + processed_memory)) # (batch, max_time, 1)
		alignment = alignment.squeeze(-1)  # (batch, max_time)
		
		return alignment


#################
# ATTENTION RNN #
#################
class AttentionRNN(nn.Module):

	def __init__(self, 
				 rnn_cell, 
				 attention_mechanism,
				 attention,
				 score_mask_value=-float("inf")):

		super(AttentionRNN, self).__init__()

		self.rnn_cell = rnn_cell
		self.attention_mechanism = attention_mechanism
		self.attention = attention
		if self.attention == 'Bahdanau':
			self.memory_layer = nn.Linear(256, 256, bias=False)
		elif self.attention == 'LocationSensitive':
			self.memory_layer = LinearNorm(256, 256, bias=False, w_init_gain='tanh')
		self.score_mask_value = score_mask_value

	def forward(self,
				query, 
				attention, 
				cell_state, 
				memory,
				attention_weights_cat=None,
				processed_memory=None, 
				mask=None, 
				memory_lengths=None):
		
		if self.attention == 'LocationSensitive' and attention_weights_cat is None:
			raise RuntimeError('Missing input: attention_weights_cat')
		if processed_memory is None:
			processed_memory = memory
		if memory_lengths is not None and mask is None:
			mask = get_mask_from_lengths(memory, memory_lengths)

		cell_input = torch.cat((query, attention), -1) # Concat input query and previous attention context
		cell_output = self.rnn_cell(cell_input, cell_state) # Feed it to RNN
		
		if self.attention == 'Bahdanau':
			alignment = self.attention_mechanism(cell_output, processed_memory) # Alignment: (batch, max_time)
		elif self.attention == 'LocationSensitive':
			alignment = self.attention_mechanism(cell_output, processed_memory, attention_weights_cat)

		if mask is not None:
			mask = mask.view(query.size(0), -1)
			alignment.data.masked_fill_(mask, self.score_mask_value)

		alignment = F.softmax(alignment, dim=1) # Normalize attention weight
		attention = torch.bmm(alignment.unsqueeze(1), memory) # Attention context vector: (batch, 1, dim)
		attention = attention.squeeze(1) # (batch, dim)

		return cell_output, attention, alignment

