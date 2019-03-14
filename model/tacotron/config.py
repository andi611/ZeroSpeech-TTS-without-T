# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ config.py ]
#   Synopsis     [ configurations ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import argparse
from multiprocessing import cpu_count


########################
# MODEL CONFIGURATIONS #
########################
class configurations(object):
	
	def __init__(self):
		self.get_audio_config()
		self.get_model_config()
		self.get_loss_config()
		self.get_dataloader_config()
		self.get_training_config()
		self.get_testing_config()

	def get_audio_config(self):
		self.num_mels = 80
		self.num_freq = 1025
		self.sample_rate = 22050
		self.frame_length_ms = 50
		self.frame_shift_ms = 12.5
		self.preemphasis = 0.97
		self.min_level_db = -100
		self.ref_level_db = 20
		self.hop_length = 250

	def get_model_config(self):
		self.embedding_dim = 256
		self.outputs_per_step = 5
		self.padding_idx = None
		self.attention = 'LocationSensitive' # or 'Bahdanau'
		self.use_mask = False

	def get_loss_config(self):
		self.prior_freq = 3000
		self.prior_weight = 0.5
		self.gate_coefficient = 0.1

	def get_dataloader_config(self):
		self.pin_memory = True
		self.num_workers = cpu_count() # or just set 2

	def get_training_config(self):
		self.batch_size = 8
		self.adam_beta1 = 0.9
		self.adam_beta2 = 0.999
		self.initial_learning_rate = 0.002
		self.decay_learning_rate = True
		self.max_epochs = 1000
		self.max_steps = 500000
		self.weight_decay = 0.0
		self.clip_thresh = 1.0
		self.checkpoint_interval = 2000

	def get_testing_config(self):
		self.max_iters = 200
		self.max_decoder_steps = 500
		self.griffin_lim_iters = 60
		self.power = 1.5 # Power to raise magnitudes to prior to Griffin-Lim

config = configurations()


###########################
# TRAINING CONFIGURATIONS #
###########################
def get_training_args():
	parser = argparse.ArgumentParser(description='training arguments')

	parser.add_argument('--ckpt_dir', type=str, default='./ckpt', help='Directory where to save model checkpoints')
	parser.add_argument('--model_name', type=str, default=None, help='Restore model from checkpoint path if name is given')
	parser.add_argument('--data_root', type=str, default='./data/meta', help='Directory that contains preprocessed model-ready features')
	parser.add_argument('--meta_text', type=str, default='meta_text.txt', help='Model-ready training transcripts')
	parser.add_argument('--log_dir', type=str, default=None, help='Directory for log summary writer to write in')
	parser.add_argument('--log_comment', type=str, default=None, help='Comment to add to the directory for log summary writer')

	args = parser.parse_args()
	return args


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
	parser = argparse.ArgumentParser(description='preprocess arguments')

	parser.add_argument('--mode', choices=['make', 'analyze', 'all'], default='all', help='what to preprocess')
	parser.add_argument('--num_workers', type=int, default=cpu_count(), help='multi-thread processing')
	parser.add_argument('--file_suffix', type=str, default='wav', help='audio filename extension')

	meta_path = parser.add_argument_group('meta_path')
	meta_path.add_argument('--meta_dir', type=str, default='./data/meta/', help='path to the model-ready training acoustic features')
	meta_path.add_argument('--meta_text', type=str, default='meta_text.txt', help='name of the model-ready training transcripts')

	input_path = parser.add_argument_group('input_path')
	input_path.add_argument('--text_input_path', type=str, default='./data/LJSpeech-1.1/metadata.csv', help='path to the original training text data')
	input_path.add_argument('--audio_input_dir', type=str, default='./data/LJSpeech-1.1/wavs/', help='path to the original training audio data')

	args = parser.parse_args()
	return args


#######################
# TEST CONFIGURATIONS #
#######################
def get_test_args():
	parser = argparse.ArgumentParser(description='testing arguments')

	parser.add_argument('--plot', action='store_true', help='whether to plot')
	parser.add_argument('--interactive', action='store_true', help='whether to test in an interactive mode')

	path_parser = parser.add_argument_group('path')
	path_parser.add_argument('--result_dir', type=str, default='./result/', help='path to output test results')
	path_parser.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='path to the directory where model checkpoints are saved')
	path_parser.add_argument('--checkpoint_name', type=str, default='checkpoint_step', help='model name prefix for checkpoint files')
	path_parser.add_argument('--model_name', type=str, default='500000', help='model step name for checkpoint files')
	path_parser.add_argument('--test_file_path', type=str, default='./data/test_transcripts.txt', help='path to the input test transcripts')
	
	args = parser.parse_args()
	return args

