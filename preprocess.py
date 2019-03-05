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
			   dset='train',
			   remake=True):
	
	if remake or not os.path.isfile(dataset_path):
		with h5py.File(dataset_path, 'w') as h5py_file:
			grps = [h5py_file.create_group('train'), h5py_file.create_group('test')]
			print('[Processor] - making training dataset...')
			make_dataset(grps, seg_len, root_dir=source_path)
			make_dataset(grps, seg_len, root_dir=target_path)
			
			print('[Processor] - making testing dataset...')
			make_dataset(grps, seg_len, root_dir=test_path, make_test=True, pad=False)

	# stage 1 training samples
	print('[Processor] - making stage 1 training samples with segment length = ', seg_len)
	make_samples(dataset_path, index_path, speaker2id_path,
				make_object='all',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

	# stage 2 training source samples
	print('[Processor] - making stage 2 training source samples with segment length = ', seg_len)
	make_samples(dataset_path, index_source_path, speaker2id_path,
				make_object='source',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()

	# stage 2 training target samples
	print('[Processor] - making stage 2 training target samples with segment length = ', seg_len)
	make_samples(dataset_path, index_target_path, speaker2id_path,
				make_object='target',
				seg_len=seg_len, 
				n_samples=n_samples, 
				dset=dset)
	print()


def make_dataset(grps, seg_len, root_dir, make_test=False, pad=True):
	
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
			speaker_id, segment_id = filename.strip().split('/')[-1].strip('.wav').split('_')
			mel_spec, lin_spec = get_spectrograms(filename)

			if pad and len(lin_spec) <= seg_len:
				mel_padding = np.zeros((seg_len - mel_spec.shape[0] + 1, mel_spec.shape[1]))
				lin_padding = np.zeros((seg_len - lin_spec.shape[0] + 1, lin_spec.shape[1]))
				mel_spec = np.concatenate((mel_spec, mel_padding), axis=0)
				lin_spec = np.concatenate((lin_spec, lin_padding), axis=0)
				print('[Processor] - processing {}: {} - padded to {}'.format(speaker_id, filename, np.shape(lin_spec)), end='\r')
			else:
				print('[Processor] - processing {}: {}'.format(speaker_id, filename), end='\r')
				
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
			raise NotImplementedError('Invalid make object!')
		print('[Sampler] - Speaker used: ', self.speaker_used)

		self.speaker2utts = {speaker : sorted(list(self.f_h5[f'{dset}/{speaker}'].keys())) for speaker in self.speaker_used}
		self.rm_too_short_utt()
		self.speaker_weight = [len(self.speaker2utts[speaker_id]) / self.total_utt for speaker_id in self.speaker_used]
		self.indexer = namedtuple('index', ['speaker', 'i', 't'])


	def get_num_utts(self):
		cnt = 0
		for speaker_id in self.speaker_used: cnt += len(self.speaker2utts[speaker_id])
		return cnt


	def rm_too_short_utt(self, limit=None):
		self.total_utt = self.get_num_utts()
		to_rm = defaultdict(lambda : [])
		if limit is None:
			limit = self.seg_len
		for speaker_id in self.speaker_used:
			for utt_id in self.speaker2utts[speaker_id]:
				if self.f_h5[f'{self.dset}/{speaker_id}/{utt_id}/lin'].shape[0] <= limit:
					to_rm[speaker_id].append(utt_id)
		for speaker_id, utt_ids in to_rm.items():
			for utt_id in utt_ids:
				self.speaker2utts[speaker_id].remove(utt_id)
		new_cnt = self.get_num_utts()
		print('[Sampler] - %i too short utterences out of a total of %i are removed.' % (self.total_utt - new_cnt, self.total_utt))

				
	def sample_utt(self, speaker_id, n_samples=1):
		# sample an utterence
		utt_ids = random.sample(self.speaker2utts[speaker_id], n_samples)
		lengths = [self.f_h5[f'{self.dset}/{speaker_id}/{utt_id}/lin'].shape[0] for utt_id in utt_ids]
		return [(utt_id, length) for utt_id, length in zip(utt_ids, lengths)]


	def rand(self, l):
		rand_idx = random.randint(0, len(l) - 1)
		return l[rand_idx] 


	def sample(self):
		speaker = np.random.choice(self.speaker_used, p=self.speaker_weight)
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
	Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
	Args:
	  sound_file: A string. The full path of a sound file.

	Returns:
	  mel: A 2d array of shape (T, n_mels) <- Transposed
	  mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
"""
def get_spectrograms(sound_file):

	
	y, sr = librosa.load(sound_file, sr=hp.sr) # Loading sound file
	y, _ = librosa.effects.trim(y) # Trimming
	y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1]) # Preemphasis

	linear = librosa.stft(y=y, # stft
						  n_fft=hp.n_fft,
						  hop_length=hp.hop_length,
						  win_length=hp.win_length)

	
	mag = np.abs(linear)  # magnitude spectrogram: (1+n_fft//2, T)

	# mel spectrogram
	mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
	mel = np.dot(mel_basis, mag)  # (n_mels, t)

	# to decibel
	mel = 20 * np.log10(np.maximum(1e-5, mel))
	mag = 20 * np.log10(np.maximum(1e-5, mag))

	# normalize
	mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
	mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

	# Transpose
	mel = mel.T.astype(np.float32)  # (T, n_mels)
	mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

	return mel, mag
