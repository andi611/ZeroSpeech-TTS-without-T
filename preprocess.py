# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess.py ]
#   Synopsis     [ pre-processing functions that parses the zerospeech2019 dataset ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import h5py
import glob
import json 
import random
import librosa
import numpy as np
from collections import namedtuple
from collections import defaultdict
from hps.hps import hp


def preprocess(source_path, 
			   target_path,
			   test_path,
			   dataset_path, 
			   index_path, 
			   index_source_path, 
			   index_target_path, 
			   speaker2id_path,
			   seg_len=128, 
			   n_samples=200000,
			   dset='train'):

	with h5py.File(dataset_path, 'w') as h5py_file:
		grps = [h5py_file.create_group('train'), h5py_file.create_group('test')]
		print('[Processor] - making training dataset...')
		make_dataset(grps, root_dir=source_path)
		make_dataset(grps, root_dir=target_path)
		
		print('[Processor] - making testing dataset...')
		make_dataset(grps, root_dir=test_path, make_test=True)

	# stage 1 training samples
	print('[Processor] - making stage 1 training samples...')
	make_samples(dataset_path, index_path, speaker2id_path,
				make_object='all',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

	# stage 2 training source samples
	print('[Processor] - making stage 2 training source samples...')
	make_samples(dataset_path, index_source_path, speaker2id_path,
				make_object='source',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

	# stage 2 training target samples
	print('[Processor] - making stage 2 training target samples...')
	make_samples(dataset_path, index_target_path, speaker2id_path,
				make_object='target',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()


def make_dataset(grps, root_dir, make_test=False):
	
	filenames = glob.glob(os.path.join(root_dir, '*_*.wav'))
	filename_groups = defaultdict(lambda : [])

	for filename in filenames:
		# divide into groups
		speaker_id, segment_id = filename.strip().split('/')[-1].strip('.wav').split('_')
		filename_groups[speaker_id].append(filename)
	
	print('Number of speakers: ', len(filename_groups))
	grp = grps[1] if make_test else grps[0]

	for speaker_id, filenames in filename_groups.items():
		for filename in filenames:
			print('[Processor] - processing {}: {}'.format(speaker_id, filename), end='\r')
			speaker_id, segment_id = filename.strip().split('/')[-1].strip('.wav').split('_')
			mel_spec, lin_spec = get_spectrograms(filename)
			grp.create_dataset('{}/{}/mel'.format(speaker_id, segment_id), data=mel_spec, dtype=np.float32)
			grp.create_dataset('{}/{}/lin'.format(speaker_id, segment_id), data=lin_spec, dtype=np.float32)
		print() 
	print()


def make_samples(h5py_path, 
				 json_path, 
				 speaker2id_path, 
				 make_object, 
				 seg_len=64, 
				 n_samples=200000, 
				 dset='train'):

	sampler = Sampler(h5py_path, dset, seg_len, speaker2id_path, make_object)
	samples = [sampler.sample()._asdict() for _ in range(n_samples)]
	with open(json_path, 'w') as f_json:
		json.dump(samples, f_json, indent=4, separators=(',', ': '))


class Sampler(object):
	def __init__(self, 
				 h5_path,
				 dset='train', 
				 seg_len=64,
				 speaker2id_path='',
				 make_object='all'):

		self.dset = dset
		self.f_h5 = h5py.File(h5_path, 'r')
		self.seg_len = seg_len
		self.speaker2id_path = speaker2id_path

		if make_object == 'all': 
			self.speaker_used = sorted(list(self.f_h5[dset].keys()))
			self.save_speaker2id()
			print('[Sampler] - Generating stage 1 training segments...')
		elif make_object == 'source':
			self.get_speaker2id()
			self.speaker_used = [s for s in sorted(list(self.f_h5[dset].keys())) if s not in ['V001', 'V002']]
			print('[Sampler] - Generating stage 2 training source segments...')
		elif make_object == 'target':
			self.get_speaker2id()
			self.speaker_used = ['V001', 'V002']
			print('[Sampler] - Generating stage 2 training target segments...')
		else:
			raise NotImplementedError()
		print('[Sampler] - Speaker used: ', self.speaker_used)

		self.speaker2utts = {speaker : sorted(list(self.f_h5[f'{dset}/{speaker}'].keys())) for speaker in self.speaker_used}
		self.rm_too_short_utt()
		self.indexer = namedtuple('index', ['speaker', 'i', 't'])


	def get_num_utts(self):
		cnt = 0
		for speaker_id in self.speaker_used: cnt += len(self.speaker2utts[speaker_id])
		return cnt


	def rm_too_short_utt(self, limit=None):
		ori_cnt = self.get_num_utts()
		to_rm = defaultdict(lambda : [])
		if limit is None:
			limit = self.seg_len * 2
		for speaker_id in self.speaker_used:
			for utt_id in self.speaker2utts[speaker_id]:
				if self.f_h5[f'{self.dset}/{speaker_id}/{utt_id}/lin'].shape[0] <= limit:
					to_rm[speaker_id].append(utt_id)
		for speaker_id, utt_ids in to_rm.items():
			for utt_id in utt_ids:
				self.speaker2utts[speaker_id].remove(utt_id)
		new_cnt = self.get_num_utts()
		print('[Sampler] - %i too short utterences out of a total of %i are removed.' % (ori_cnt - new_cnt, ori_cnt))

				
	def sample_utt(self, speaker_id, n_samples=1):
		# sample an utterence
		utt_ids = random.sample(self.speaker2utts[speaker_id], n_samples)
		lengths = [self.f_h5[f'{self.dset}/{speaker_id}/{utt_id}/lin'].shape[0] for utt_id in utt_ids]
		return [(utt_id, length) for utt_id, length in zip(utt_ids, lengths)]


	def rand(self, l):
		rand_idx = random.randint(0, len(l) - 1)
		return l[rand_idx] 


	def sample(self):
		speaker, = random.sample(self.speaker_used, 1)
		speaker_idx = self.speaker2id[speaker]
		(utt_id, utt_len), = self.sample_utt(speaker, 1)
		t = random.randint(0, utt_len - self.seg_len)  
		index_tuple = self.indexer(speaker=speaker_idx, i=f'{speaker}/{utt_id}', t=t)
		return index_tuple


	def save_speaker2id(self):
		self.speaker2id = {speaker:i for i, speaker in enumerate(self.speaker_used)}
		with open(self.speaker2id_path, 'w') as file:
			file.write(json.dumps(self.speaker2id))

	def get_speaker2id(self):
		with open(self.speaker2id_path, 'r') as f_json:
			self.speaker2id = json.load(f_json)


"""
	Extracts melspectrogram and log magnitude from given `sound_file`.
	Args:
		sound_file: A string. Full path of a sound file.
	Returns:
		Transposed S: A 2d array. A transposed melspectrogram with shape of (T, n_mels)
		Transposed magnitude: A 2d array.Has shape of (T, 1+hp.n_fft//2)
"""
def get_spectrograms(sound_file): 

	y, sr = librosa.load(sound_file, sr=hp.sr) # Loading sound file
	y, _ = librosa.effects.trim(y) # Trimming
	D = librosa.stft(y=y, # stft. D: (1+n_fft//2, T)
					 n_fft=hp.n_fft, 
					 hop_length=hp.hop_length, 
					 win_length=hp.win_length) 
	
	magnitude = np.abs(D) # magnitude spectrogram: (1+n_fft/2, T)
	power = magnitude**2 # power spectrogram: (1+n_fft/2, T) 
	S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels) # mel spectrogram: (n_mels, T)

	return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (T, n_mels), (T, 1+n_fft/2)

