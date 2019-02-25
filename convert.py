# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ convert.py ]
#   Synopsis     [ testing functions for voice conversion ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os 
import h5py
import json
import copy
import torch
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy import signal
from scipy.io.wavfile import write
from torch.autograd import Variable
from preprocess import get_spectrograms
from trainer import Trainer
from hps.hps import hp, Hps


def griffin_lim(spectrogram): # Applies Griffin-Lim's raw.
	
	def _invert_spectrogram(spectrogram): # spectrogram: [f, t]
		return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

	X_best = copy.deepcopy(spectrogram)
	for i in range(hp.n_iter):
		X_t = _invert_spectrogram(X_best)
		est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
		phase = est / np.maximum(1e-8, np.abs(est))
		X_best = spectrogram * phase
	X_t = _invert_spectrogram(X_best)
	y = np.real(X_t)
	return y


def spectrogram2wav(mag): # Generate wave file from spectrogram
	mag = mag.T # transpose
	mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db # de-noramlize
	mag = np.power(10.0, mag * 0.05) # to amplitude
	wav = griffin_lim(mag) # wav reconstruction
	wav = signal.lfilter([1], [1, -hp.preemphasis], wav) # de-preemphasis
	wav, _ = librosa.effects.trim(wav) # trim
	return wav.astype(np.float32)


def sp2wav(sp): 
	exp_sp = sp
	wav_data = spectrogram2wav(exp_sp)
	return wav_data


def get_world_param(f_h5, src_speaker, utt_id, tar_speaker, tar_speaker_id, trainer, dset='test', gen=True):
	mc = f_h5[f'{dset}/{src_speaker}/{utt_id}/norm_mc'][()]
	converted_mc = convert_x(mc, tar_speaker_id, trainer, gen=gen)
	#converted_mc = mc
	mc_mean = f_h5[f'train/{tar_speaker}'].attrs['mc_mean']
	mc_std = f_h5[f'train/{tar_speaker}'].attrs['mc_std']
	converted_mc = converted_mc * mc_std + mc_mean
	log_f0 = f_h5[f'{dset}/{src_speaker}/{utt_id}/log_f0'][()]
	src_mean = f_h5[f'train/{src_speaker}'].attrs['f0_mean']
	src_std = f_h5[f'train/{src_speaker}'].attrs['f0_std']
	tar_mean = f_h5[f'train/{tar_speaker}'].attrs['f0_mean']
	tar_std = f_h5[f'train/{tar_speaker}'].attrs['f0_std']
	index = np.where(log_f0 > 1e-10)[0]
	log_f0[index] = (log_f0[index] - src_mean) * tar_std / src_std + tar_mean
	log_f0[index] = np.exp(log_f0[index])
	f0 = log_f0
	ap = f_h5[f'{dset}/{src_speaker}/{utt_id}/ap'][()]
	converted_mc = converted_mc[:ap.shape[0]]
	sp = pysptk.conversion.mc2sp(converted_mc, alpha=0.41, fftlen=1024)
	return f0, sp, ap


def synthesis(f0, sp, ap, sr=16000):
	y = pw.synthesize(f0.astype(np.float64), sp.astype(np.float64), ap.astype(np.float64), sr, pw.default_frame_period)
	return y


def convert_x(x, c, trainer, enc_only):
	c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
	tensor = torch.from_numpy(np.expand_dims(x, axis=0))
	tensor = tensor.type(torch.FloatTensor)
	converted = trainer.test_step(tensor, c_var, enc_only=enc_only)
	converted = converted.squeeze(axis=0).transpose((1, 0))
	return converted


def get_trainer(hps_path, model_path, targeted_G, one_hot):
	HPS = Hps(hps_path)
	hps = HPS.get_tuple()
	trainer = Trainer(hps, None, targeted_G, one_hot)
	trainer.load_model(model_path, model_all=False)
	return trainer


def convert_all_sp(trainer,
				   h5_path, 
				   src_speaker, 
				   tar_speaker, 
				   enc_only=True, 
				   dset='test', 
				   speaker2id={},
				   result_dir=''):
	with h5py.File(h5_path, 'r') as f_h5:
		for utt_id in f_h5[f'{dset}/{src_speaker}']:
			try:
				sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
				converted_x = convert_x(sp, speaker2id[tar_speaker], trainer, enc_only=enc_only)
				wav_data = sp2wav(converted_x)
				wav_path = os.path.join(result_dir, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
				sf.write(wav_path, wav_data, 16000, 'PCM_24')
			except RuntimeError:
				print('[Tester] - Unable to process \"{}\" of speaker {}: '.format(utt_id, f_h5[f'{dset}/{src_speaker}']))


def convert_all_mc(trainer,
				   h5_path, 
				   src_speaker, 
				   tar_speaker, 
				   gen=False, 
				   dset='test', 
				   speaker2id={},
				   result_dir=''):
	with h5py.File(h5_path, 'r') as f_h5:
		for utt_id in f_h5[f'{dset}/{src_speaker}']:
			f0, sp, ap = get_world_param(f_h5, src_speaker, utt_id, tar_speaker, tar_speaker_id=speaker2id[tar_speaker], trainer=trainer, dset='test', gen=gen)
			wav_data = synthesis(f0, sp, ap)
			wav_path = os.path.join(result_dir, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
			sf.write(wav_path, wav_data, 16000, 'PCM_24')


def test(trainer, data_path, speaker2id_path, result_dir, enc_only, flag):

	f_h5 = h5py.File(data_path, 'r')
	
	print('[Tester] - Testing on the {}ing set...'.format(flag))
	if flag == 'test':
		source_speakers = sorted(list(f_h5['test'].keys()))
	elif flag == 'train':
		source_speakers = [s for s in sorted(list(f_h5['train'].keys())) if s[0] == 'S']
	target_speakers = [s for s in sorted(list(f_h5['train'].keys())) if s[0] == 'V']
	print('[Tester] - Source speakers: %i, Target speakers: %i' % (len(source_speakers), len(target_speakers)))

	with open(speaker2id_path, 'r') as f_json:
		speaker2id = json.load(f_json)

	print('[Tester] - Converting all testing utterances from source speakers to target speakers, this may take a while...')
	for speaker_S in tqdm(source_speakers):
		for speaker_T in target_speakers:
			assert speaker_S != speaker_T
			dir_path = os.path.join(result_dir, f'p{speaker_S}_p{speaker_T}')
			os.makedirs(dir_path, exist_ok=True)

			convert_all_sp(trainer,
						   data_path, 
						   speaker_S, 
						   speaker_T,
						   enc_only=enc_only,
						   dset=flag,
						   speaker2id=speaker2id,
						   result_dir=dir_path)


def test_single(trainer, seg_len, speaker2id_path, result_dir, enc_only, s_speaker, t_speaker):

	with open(speaker2id_path, 'r') as f_json:
		speaker2id = json.load(f_json)

	if s_speaker == 'S015':
		filename = './data/english/train/unit/S015_0361841101.wav' 
	elif s_speaker == 'S119':
		filename = './data/english/train/unit/S119_1561145062.wav' 
	else:
		raise NotImplementedError('Please modify path manually!')
	
	_, spec = get_spectrograms(filename)
	results = []

	for idx in range(0, len(spec), seg_len):
		try:
			spec_frag = spec[idx:idx+seg_len]
		except: 
			spec_frag = spec[idx:-1]
		spec_expand = np.expand_dims(spec_frag, axis=0)
		spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
		c = Variable(torch.from_numpy(np.array([speaker2id[t_speaker]]))).cuda()
		result = trainer.test_step(spec_tensor, c, enc_only=enc_only, verbose=True)
		result = result.squeeze(axis=0).transpose((1, 0))
		results.append(result)

	results = np.concatenate(results, axis=0)
	wav_data = spectrogram2wav(results)

	write(os.path.join(result_dir, 'result.wav'), rate=16000, data=wav_data)
	print('Testing on source speaker {} and target speaker {}, output shape: {}'.format(s_speaker, t_speaker, results.shape))

