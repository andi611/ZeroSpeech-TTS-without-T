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


def synthesis(f0, sp, ap, sr=16000):
	y = pw.synthesize(f0.astype(np.float64), sp.astype(np.float64), ap.astype(np.float64), sr, pw.default_frame_period)
	return y


def convert_x(x, c, trainer, enc_only, verbose=False):
	c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
	tensor = torch.from_numpy(np.expand_dims(x, axis=0)).type(torch.FloatTensor)
	converted, enc = trainer.test_step(tensor, c_var, enc_only=enc_only, verbose=verbose)
	converted = converted.squeeze(axis=0).transpose((1, 0))
	enc = enc.squeeze(axis=0).transpose((1, 0))
	return converted, enc


def get_trainer(hps_path, model_path, targeted_G, one_hot):
	HPS = Hps(hps_path)
	hps = HPS.get_tuple()
	trainer = Trainer(hps, None, targeted_G, one_hot)
	trainer.load_model(model_path, model_all=False)
	return trainer


def convert(trainer,
			seg_len,
			src_speaker_spec, 
			tar_speaker,
			utt_id,
			speaker2id,
			result_dir,
			enc_only=True,
			save=True): 
	
	if len(src_speaker_spec) > seg_len:
		converted_results = []
		encodings = []
		for idx in range(0, len(src_speaker_spec), seg_len):
			try: spec_frag = src_speaker_spec[idx:idx+seg_len]
			except: spec_frag = src_speaker_spec[idx:-1]
			converted_x, enc = convert_x(spec_frag, speaker2id[tar_speaker], trainer, enc_only=enc_only)
			converted_results.append(converted_x)
			encodings.append(enc)

		converted_results = np.concatenate(converted_results, axis=0)
		encodings = np.concatenate(encodings, axis=0)

		wav_data = spectrogram2wav(converted_results)
		if save:
			wav_path = os.path.join(result_dir, f'{tar_speaker}_{utt_id}.wav')
			sf.write(wav_path, wav_data, hp.sr, 'PCM_24')
		else:
			return wav_data, encodings
	else:
		print('[Tester] - Unable to process \"{}\" of speaker {}: '.format(utt_id, f_h5[f'{dset}/{src_speaker}']))


def test_from_list(trainer, seg_len, synthesis_list, data_path, speaker2id_path, result_dir, enc_only, flag='test'):
	
	with open(speaker2id_path, 'r') as f_json:
		speaker2id = json.load(f_json)

	feeds = []
	with open(synthesis_list, 'r') as f:
		file = f.read()
		for line in file:
			line = line.split()
			feeds.append({'s_id' : line[0].split('/')[1].split('_')[0],
						  'utt_id' : line[0].split('/')[1].split('_')[1], 
						  't_id' : line[1], })

	print('[Tester] - Number of files to be resynthesize: ', len(feeds))
	dir_path = os.path.join(result_dir, f'{flag}/')
	os.makedirs(dir_path, exist_ok=True)

	with h5py.File(data_path, 'r') as f_h5:
		for feed in feeds:
			convert(trainer,
					seg_len,
					src_speaker_spec=f_h5[f"test/{feed['s_id']}/{feed['utt_id']}/lin"][()], 
					tar_speaker=feed['t_id'],
					utt_id=feed['utt_id'],
					speaker2id=speaker2id,
					result_dir=dir_path,
					enc_only=enc_only)



def cross_test(trainer, seg_len, data_path, speaker2id_path, result_dir, enc_only, flag):

	with h5py.File(data_path, 'r') as f_h5:

		with open(speaker2id_path, 'r') as f_json:
			speaker2id = json.load(f_json)
		
		if flag == 'test':
			source_speakers = sorted(list(f_h5['test'].keys()))
		elif flag == 'train':
			source_speakers = [s for s in sorted(list(f_h5['train'].keys())) if s[0] == 'S']
		target_speakers = [s for s in sorted(list(f_h5['train'].keys())) if s[0] == 'V']

		print('[Tester] - Testing on the {}ing set...'.format(flag))
		print('[Tester] - Source speakers: %i, Target speakers: %i' % (len(source_speakers), len(target_speakers)))
		print('[Tester] - Converting all testing utterances from source speakers to target speakers, this may take a while...')
	
		for src_speaker in tqdm(source_speakers):
			for tar_speaker in target_speakers:
				assert src_speaker != tar_speaker
				dir_path = os.path.join(result_dir, f'{src_speaker}_to_{tar_speaker}')
				os.makedirs(dir_path, exist_ok=True)

				for utt_id in f_h5[f'test/{src_speaker}']:
					src_speaker_spec = f_h5[f'test/{src_speaker}/{utt_id}/lin'][()]
					convert(trainer,
							seg_len,
							src_speaker_spec, 
							tar_speaker,
							utt_id=utt_id,
							speaker2id=speaker2id,
							result_dir=dir_path,
							enc_only=enc_only)


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
	wav_data, encodings = convert(trainer,
								  seg_len,
								  src_speaker_spec=spec, 
								  tar_speaker=t_speaker,
								  utt_id='',
								  speaker2id=speaker2id,
								  result_dir=result_dir,
								  enc_only=enc_only,
								  save=False)

	sf.write(os.path.join(result_dir, 'result.wav'), wav_data, hp.sr, 'PCM_24')
	with open(os.path.join(result_dir, 'result.txt'), 'w') as file:
    	for enc in encodings: file.write(enc + '\n')
	print('Testing on source speaker {} and target speaker {}, output shape: {}'.format(s_speaker, t_speaker, wav_data.shape))

