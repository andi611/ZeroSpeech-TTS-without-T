# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ hps.py ]
#   Synopsis     [ hyperparams ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import json
from collections import namedtuple


class processing_hyperparams(object):
	def __init__(self):
		self.max_duration = 10.0

		# signal processing
		self.sr = 16000 # Sample rate.
		self.n_fft = 1024 # fft points (samples)
		self.frame_shift = 0.0125 # seconds
		self.frame_length = 0.05 # seconds
		self.hop_length = int(self.sr*self.frame_shift) # samples.
		self.win_length = int(self.sr*self.frame_length) # samples.
		self.n_mels = 80 # Number of Mel banks to generate
		self.power = 1.2 # Exponent for amplifying the predicted magnitude
		self.n_iter = 300 # Number of inversion iterations
		self.preemphasis = .97 # or None
		self.max_db = 100
		self.ref_db = 20
		self.prior_freq = 3000
		self.prior_weight = 0.5
hp = processing_hyperparams()


class Hps(object):
	def __init__(self, path=None):
		self.hps = namedtuple('hps', [
			'g_mode',
			'enc_mode',
			'load_model_list',
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
			'enc_size',
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
			'tacotron_iters',
			'tclf_iters',
			'max_to_keep',
			]
		)
		if not path is None:
			self.load(path)
		else:
			print('Using default parameters since no .json file is provided.')
			default = \
				['enhanced', 'continues', 1e-4, 1, 1e-4, 0, 0, 0, 10, 0.01, 0.5, 0.1, 5, 5, 128, 400000, 1024, 1024, 102, 2, 5, 0, 32, 50000, 5000, 5000, 30000, 60000, 10]
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

