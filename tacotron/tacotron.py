# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ tacotron.py ]
#   Synopsis     [ Tacotron model in Pytorch ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn
from torch.autograd import Variable
from model.attention import BahdanauAttention, LocationSensitiveAttention
from model.attention import AttentionRNN
from model.loss import get_rnn_mask_from_lengths


##########
# PRENET #
##########
class Prenet(nn.Module):
	
	def __init__(self, in_dim, sizes=[256, 128]):
		
		super(Prenet, self).__init__()
		in_sizes = [in_dim] + sizes[:-1]
		self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (in_size, out_size) in zip(in_sizes, sizes)])
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

	def forward(self, inputs):
		for linear in self.layers:
			inputs = self.dropout(self.relu(linear(inputs)))
		return inputs


#####################
# BATCH NORM CONV1D #
#####################
class BatchNormConv1d(nn.Module):
	
	def __init__(self, in_dim, out_dim, kernel_size, stride, padding, activation=None):

		super(BatchNormConv1d, self).__init__()
		self.conv1d = nn.Conv1d(in_dim, 
								out_dim,
								kernel_size=kernel_size,
								stride=stride, 
								padding=padding, 
								bias=False)
		
		self.bn = nn.BatchNorm1d(out_dim, momentum=0.99, eps=1e-3) # Following tensorflow's default parameters
		self.activation = activation

	def forward(self, x):
		x = self.conv1d(x)
		if self.activation is not None:
			x = self.activation(x)
		return self.bn(x)


###########
# HIGHWAY #
###########
class Highway(nn.Module):
	
	def __init__(self, in_size, out_size):

		super(Highway, self).__init__()
		self.H = nn.Linear(in_size, out_size)
		self.H.bias.data.zero_()
		self.T = nn.Linear(in_size, out_size)
		self.T.bias.data.fill_(-1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, inputs):
		H = self.relu(self.H(inputs))
		T = self.sigmoid(self.T(inputs))
		return H * T + inputs * (1.0 - T)


###############
# CBHG MODULE #
###############
"""
	CBHG module: a recurrent neural network composed of:
		- 1-d convolution banks
		- Highway networks + residual connections
		- Bidirectional gated recurrent units
"""
class CBHG(nn.Module):

	def __init__(self, in_dim, K=16, projections=[128, 128]):
		
		super(CBHG, self).__init__()
		self.in_dim = in_dim
		self.relu = nn.ReLU()
		self.conv1d_banks = nn.ModuleList(
			[BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
							 padding=k // 2, activation=self.relu)
			 for k in range(1, K + 1)])
		self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

		in_sizes = [K * in_dim] + projections[:-1]
		activations = [self.relu] * (len(projections) - 1) + [None]
		self.conv1d_projections = nn.ModuleList(
			[BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
							 padding=1, activation=ac)
			 for (in_size, out_size, ac) in zip(
				 in_sizes, projections, activations)])

		self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
		self.highways = nn.ModuleList(
			[Highway(in_dim, in_dim) for _ in range(4)])

		self.gru = nn.GRU(
			in_dim, in_dim, 1, batch_first=True, bidirectional=True)

	def forward(self, inputs, input_lengths=None):
		
		x = inputs # (B, T_in, in_dim)

		if x.size(-1) == self.in_dim: # Needed to perform conv1d on time-axis: (B, in_dim, T_in)
			x = x.transpose(1, 2)

		T = x.size(-1)

		x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1) # (B, in_dim*K, T_in) -> Concat conv1d bank outputs
		assert x.size(1) == self.in_dim * len(self.conv1d_banks)
		x = self.max_pool1d(x)[:, :, :T]

		for conv1d in self.conv1d_projections:
			x = conv1d(x)

		x = x.transpose(1, 2) # (B, T_in, in_dim) -> Back to the original shape

		if x.size(-1) != self.in_dim:
			x = self.pre_highway(x)

		
		x += inputs # Residual connection
		for highway in self.highways:
			x = highway(x)

		if input_lengths is not None:
			x = nn.utils.rnn.pack_padded_sequence(
				x, input_lengths, batch_first=True)

		outputs, _ = self.gru(x) # (B, T_in, in_dim*2)

		if input_lengths is not None:
			outputs, _ = nn.utils.rnn.pad_packed_sequence(
				outputs, batch_first=True)

		return outputs


###########
# ENCODER #
###########
class Encoder(nn.Module):
	
	def __init__(self, in_dim):
		
		super(Encoder, self).__init__()
		self.prenet = Prenet(in_dim, sizes=[256, 128])
		self.cbhg = CBHG(128, K=16, projections=[128, 128])

	def forward(self, inputs, input_lengths=None):
		inputs = self.prenet(inputs)
		return self.cbhg(inputs, input_lengths)


###########
# DECODER #
###########
class Decoder(nn.Module):
	
	def __init__(self, in_dim, r, attention):
		
		super(Decoder, self).__init__()
		self.in_dim = in_dim
		self.r = r
		self.prenet = Prenet(in_dim * r, sizes=[256, 128])
		# (prenet_out + attention context) -> output
		
		self.attention = attention
		if self.attention == 'Bahdanau':
			self.attention_rnn = AttentionRNNh(nn.GRUCell(256 + 128, 256), 
											   BahdanauAttention(256), 
											   attention='Bahdanau')
		elif self.attention == 'LocationSensitive':
			self.attention_rnn = AttentionRNN(nn.GRUCell(256 + 128, 256), 
											  LocationSensitiveAttention(256),
											  attention='LocationSensitive')
		else: raise NotImplementedError
		
		self.project_to_decoder_in = nn.Linear(512, 256)
		self.decoder_rnns = nn.ModuleList([nn.GRUCell(256, 256) for _ in range(2)])

		self.proj_to_mel = nn.Linear(256, in_dim * self.r)
		self.proj_to_gate = nn.Linear(256, 1 * self.r)
		self.sigmoid = nn.Sigmoid()

		self.max_decoder_steps = 800


	def initialize_decoder_states(self, encoder_outputs, processed_memory, memory_lengths):
		B = encoder_outputs.size(0)
		MAX_TIME = encoder_outputs.size(1)
		
		self.initial_input = Variable(encoder_outputs.data.new(B, self.in_dim * self.r).zero_()) # go frames

		self.attention_rnn_hidden = Variable(encoder_outputs.data.new(B, 256).zero_()) # Init decoder states
		self.decoder_rnn_hiddens = [Variable(encoder_outputs.data.new(B, 256).zero_()) for _ in range(len(self.decoder_rnns))]
		self.current_attention = Variable(encoder_outputs.data.new(B, 256).zero_())

		if self.attention == 'LocationSensitive':
			self.attention_weights = Variable(encoder_outputs.data.new(B, MAX_TIME).zero_())
			self.attention_weights_cum = Variable(encoder_outputs.data.new(B, MAX_TIME).zero_())

		if memory_lengths is not None:
			self.mask = get_rnn_mask_from_lengths(processed_memory, memory_lengths)
		else:
			self.mask = None


	"""
		Decoder forward step.

		If decoder inputs are not given (e.g., at testing time), as noted in Tacotron paper, greedy decoding is adapted.

		Args:
			encoder_outputs: Encoder outputs. (B, T_encoder, dim)
			inputs: Decoder inputs. i.e., mel-spectrogram. If None (at eval-time), decoder outputs are used as decoder inputs.
			memory_lengths: Encoder output (memory) lengths. If not None, used for attention masking.
	"""
	def forward(self, encoder_outputs, inputs=None, memory_lengths=None):

		greedy = inputs is None # Run greedy decoding if inputs is None

		if inputs is not None:
			if inputs.size(-1) == self.in_dim:
				inputs = inputs.view(encoder_outputs.size(0), inputs.size(1) // self.r, -1)  # Grouping multiple frames if necessary
			assert inputs.size(-1) == self.in_dim * self.r
			T_decoder = inputs.size(1)
			inputs = inputs.transpose(0, 1) # Time first (T_decoder, B, in_dim)
		

		processed_memory = self.attention_rnn.memory_layer(encoder_outputs)
		self.initialize_decoder_states(encoder_outputs, processed_memory, memory_lengths)
			
		t = 0
		gates = []
		outputs = []
		alignments = []
		current_input = self.initial_input
		
		while True:
			if t > 0: current_input = outputs[-1] if greedy else inputs[t - 1]
			
			current_input = self.prenet(current_input) # Prenet

			if self.attention == 'Bahdanau':
				self.attention_rnn_hidden, self.current_attention, alignment = self.attention_rnn(current_input, 
																								  self.current_attention, 
																								  self.attention_rnn_hidden,
																								  encoder_outputs, 
																								  processed_memory=processed_memory, 
																								  mask=self.mask)
			elif self.attention == 'LocationSensitive':
				self.attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
				
				self.attention_rnn_hidden, self.current_attention, alignment = self.attention_rnn(current_input, 
																								  self.current_attention, 
																								  self.attention_rnn_hidden, 
																								  encoder_outputs, 
																								  self.attention_weights_cat, 
																								  processed_memory=processed_memory,
																								  mask=self.mask)
				self.attention_weights_cum += self.attention_weights
			
			decoder_input = self.project_to_decoder_in(torch.cat((self.attention_rnn_hidden, self.current_attention), -1)) # Concat RNN output and attention context vector

			# Pass through the decoder RNNs
			for idx in range(len(self.decoder_rnns)):
				self.decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](decoder_input, self.decoder_rnn_hiddens[idx])				
				decoder_input = self.decoder_rnn_hiddens[idx] + decoder_input # Residual connectinon

			output = decoder_input
			gate = self.sigmoid(self.proj_to_gate(output)).squeeze()
			output = self.proj_to_mel(output)

			outputs += [output]
			alignments += [alignment]
			gates += [gate]

			t += 1

			# testing 
			if greedy:
				if t > 1 and is_end_of_gates(gate):
					print('Terminated by gate!')
					break
				elif t > 1 and is_end_of_frames(output):
					print('Terminated by silent frames!')
					break
				elif t > self.max_decoder_steps:
					print('Warning! doesn\'t seems to be converged')
					break
			# training
			else:
				if t >= T_decoder:
					break

		assert greedy or len(outputs) == T_decoder
		
		# Back to batch first: (T_out, B) -> (B, T_out)
		alignments = torch.stack(alignments).transpose(0, 1)
		outputs = torch.stack(outputs).transpose(0, 1).contiguous()
		gates = torch.stack(gates).transpose(0, 1).contiguous()

		return outputs, alignments, gates


def is_end_of_gates(gate, thd=0.5):
	return (gate.data >= thd).all()


def is_end_of_frames(output, eps=0.2):
	return (output.data <= eps).all()


############
# TACOTRON #
############
class Tacotron(nn.Module):
	
	def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
				 r=5, padding_idx=None, attention='Bahdanau', use_mask=False):
		
		super(Tacotron, self).__init__()
		self.mel_dim = mel_dim
		self.linear_dim = linear_dim
		self.use_mask = use_mask
		
		self.embedding = nn.Embedding(n_vocab, embedding_dim, padding_idx=padding_idx)
		self.embedding.weight.data.normal_(0, 0.3) # Trying smaller std
		
		self.encoder = Encoder(embedding_dim)
		self.decoder = Decoder(mel_dim, r, attention)

		self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
		self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

	def forward(self, inputs, targets=None, input_lengths=None):
		# inputs shape: (B, enc_size, T)
		# targets shape: (B, in_dim, T)
		B = inputs.size(0)

		inputs = self.embedding(inputs)
		
		encoder_outputs = self.encoder(inputs, input_lengths) # (B, T', in_dim)

		if self.use_mask: memory_lengths = input_lengths
		else: memory_lengths = None
		
		mel_outputs, alignments, _ = self.decoder(encoder_outputs, targets, memory_lengths=memory_lengths) # (B, T', mel_dim*r)

		# Post net processing below

		mel_outputs = mel_outputs.view(B, -1, self.mel_dim) # Reshape: (B, T, mel_dim)

		linear_outputs = self.postnet(mel_outputs)
		linear_outputs = self.last_linear(linear_outputs)

		return mel_outputs, linear_outputs, alignments
