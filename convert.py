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
import glob
import h5py
import json
import copy
import torch
import librosa
import numpy as np
import soundfile as sf
import speech_recognition as sr
from jiwer import wer
from tqdm import tqdm
from scipy import signal
from trainer import Trainer
from hps.hps import hp, Hps
from torch.autograd import Variable
from preprocess import get_spectrograms


############
# CONSTANT #
############
MIN_LEN = 9


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


def encode_x(x, trainer):
	tensor = torch.from_numpy(np.expand_dims(x, axis=0)).type(torch.FloatTensor)
	enc = trainer.encoder_test_step(tensor)
	enc = enc.squeeze(axis=0).transpose((1, 0))
	return enc


def get_trainer(hps_path, model_path, g_mode, enc_mode):
	HPS = Hps(hps_path)
	hps = HPS.get_tuple()
	global MIN_LEN
	MIN_LEN = MIN_LEN if hps.enc_mode != 'gumbel_t' else hps.seg_len
	trainer = Trainer(hps, None, g_mode, enc_mode)
	trainer.load_model(model_path, load_model_list = hps.load_model_list)
	return trainer


def asr(fname):
	r = sr.Recognizer()
	with sr.WavFile(fname) as source:
		audio = r.listen(source)
	text = r.recognize_google(audio, language = 'en')
	return text


def compare_asr(s_wav, t_wav):
	try:
		gt = asr(s_wav)
		recog = asr(t_wav)
		err_result = wer(gt, recog), wer(' '.join([c for c in gt if c != ' ']), ' '.join([c for c in recog if c != ' ']))
	except sr.UnknownValueError:
		err_result = [1., 1.]
	except:
		err_result = [-1., -1.]
	return err_result


def write_encoding(path, encodings):
	with open(path, 'w') as file:
		for enc in encodings:
			for i, e in enumerate(enc):
				file.write(str(int(e)) + (' ' if i < len(enc)-1 else ''))
			file.write('\n')


def convert(trainer,
			seg_len,
			src_speaker_spec, 
			src_speaker,
			tar_speaker,
			utt_id,
			speaker2id,
			result_dir,
			enc_only=True,
			save=['wav', 'enc']): 
	
	# pad spec to minimum len
	PADDED = False
	if len(src_speaker_spec) < MIN_LEN:
		padding = np.zeros((MIN_LEN - src_speaker_spec.shape[0], src_speaker_spec.shape[1]))
		src_speaker_spec = np.concatenate((src_speaker_spec, padding), axis=0)
		PADDED = True
		
	if len(src_speaker_spec) <= seg_len:
		converted_results, encodings = convert_x(src_speaker_spec, speaker2id[tar_speaker], trainer, enc_only=enc_only)
		if PADDED: 
			encodings = encodings[:MIN_LEN//8] # truncate the encoding of zero paddings

	else:
		converted_results = []
		encodings = []
		for idx in range(0, len(src_speaker_spec), seg_len):
			if idx + (seg_len*2) > len(src_speaker_spec):
				spec_frag = src_speaker_spec[idx:-1]
			else:
				spec_frag = src_speaker_spec[idx:idx+seg_len]

			if len(spec_frag) >= seg_len:
				converted_x, enc = convert_x(spec_frag, speaker2id[tar_speaker], trainer, enc_only=enc_only)
				converted_results.append(converted_x)
				encodings.append(enc)
			elif idx == 0:
				raise RuntimeError('Please check if input is too short!')

		converted_results = np.concatenate(converted_results, axis=0)
		encodings = np.concatenate(encodings, axis=0)

	wav_data = spectrogram2wav(converted_results)
	if len(save) != 0:
		if 'wav' in save: 
			wav_path = os.path.join(result_dir, f'{tar_speaker}_{utt_id}.wav')
			sf.write(wav_path, wav_data, hp.sr, 'PCM_16')
		if 'enc' in save: 
			enc_path = os.path.join(result_dir, f'{src_speaker}_{utt_id}.txt')
			write_encoding(enc_path, encodings)
		return wav_path, len(converted_results)
	else:
		return wav_data, encodings


def encode(src_speaker_spec, trainer, seg_len, s_speaker, utt_id, result_dir=None, save=True):
	if save:
		assert result_dir != None

	# pad spec to minimum len
	PADDED = False
	if len(src_speaker_spec) < MIN_LEN:
		padding = np.zeros((MIN_LEN - src_speaker_spec.shape[0], src_speaker_spec.shape[1]))
		src_speaker_spec = np.concatenate((src_speaker_spec, padding), axis=0)
		PADDED = True
		
	if len(src_speaker_spec) <= seg_len:
		encodings = encode_x(src_speaker_spec, trainer)
		if PADDED: 
			encodings = encodings[:MIN_LEN//8] # truncate the encoding of zero paddings

	else:
		encodings = []
		for idx in range(0, len(src_speaker_spec), seg_len):
			if idx + (seg_len*2) > len(src_speaker_spec):
				spec_frag = src_speaker_spec[idx:-1]
			else:
				spec_frag = src_speaker_spec[idx:idx+seg_len]

			if len(spec_frag) >= seg_len:
				enc = encode_x(spec_frag, trainer)
				encodings.append(enc)
			elif idx == 0:
				raise RuntimeError('Please check if input is too short!')

		encodings = np.concatenate(encodings, axis=0)

	if save:
		enc_path = os.path.join(result_dir, f"{s_speaker}_{utt_id}.txt")
		write_encoding(enc_path, encodings)
	else:
		return encodings


def test_from_list(trainer, seg_len, synthesis_list, data_path, speaker2id_path, result_dir, enc_only, flag='test'):
	
	with open(speaker2id_path, 'r') as f_json:
		speaker2id = json.load(f_json)

	feeds = []
	with open(synthesis_list, 'r') as f:
		file = f.readlines()
		for line in file:
			line = line.split('\n')[0].split(' ')
			feeds.append({'s_id' : line[0].split('/')[1].split('_')[0],
						  'utt_id' : line[0].split('/')[1].split('_')[1], 
						  't_id' : line[1], })

	print('[Tester] - Number of files to be resynthesize: ', len(feeds))
	dir_path = os.path.join(result_dir, f'{flag}/')
	os.makedirs(dir_path, exist_ok=True)

	err_results = []
	with h5py.File(data_path, 'r') as f_h5:
		for feed in tqdm(feeds):
			conv_audio, n_frames = convert(trainer,
								 		   seg_len,
								 		   src_speaker_spec=f_h5[f"test/{feed['s_id']}/{feed['utt_id']}/lin"][()], 
								 		   src_speaker=feed['s_id'],
								 		   tar_speaker=feed['t_id'],
								 		   utt_id=feed['utt_id'],
								 		   speaker2id=speaker2id,
								 		   result_dir=dir_path,
								 		   enc_only=enc_only,
								 		   save=['wav'])
			n_frames = len(f_h5[f"test/{feed['s_id']}/{feed['utt_id']}/lin"][()])
			if hp.frame_shift * (n_frames - 1) + hp.frame_length >= 3.0:
				orig_audio = spectrogram2wav(f_h5[f"test/{feed['s_id']}/{feed['utt_id']}/lin"][()])
				sf.write('orig_audio.wav', orig_audio, hp.sr, 'PCM_16')
				err_results.append(compare_asr(s_wav='orig_audio.wav', t_wav=conv_audio))
				os.remove(path='orig_audio.wav')

	err_mean = np.mean(err_results, axis=0)
	print('WERR: {:.3f}  CERR: {:.3f}, computed over {} samples'.format(err_mean[0], err_mean[1], len(err_results)))


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
	elif s_speaker == 'S130':
		filename = './data/english/test/S130_3516588097.wav' 
	elif s_speaker == 'S089':
		filename = './data/english/test/S089_1810826781.wav' 
	else:
		raise NotImplementedError('Please modify path manually!')
	
	_, spec = get_spectrograms(filename)
	wav_data, encodings = convert(trainer,
								  seg_len,
								  src_speaker_spec=spec, 
								  src_speaker=s_speaker,
								  tar_speaker=t_speaker,
								  utt_id='',
								  speaker2id=speaker2id,
								  result_dir=result_dir,
								  enc_only=enc_only,
								  save=[])

	sf.write(os.path.join(result_dir, 'result.wav'), wav_data, hp.sr, 'PCM_16')
	write_encoding(os.path.join(result_dir, 'result.txt'), encodings)

	err_result = compare_asr(filename, os.path.join(result_dir, 'result.wav'))

	print('Testing on source speaker {} and target speaker {}, output shape: {}'.format(s_speaker, t_speaker, wav_data.shape))
	print('Comparing ASR result - WERR: {:.3f}  CERR: {:.3f}'.format(err_result[0], err_result[1]))


def test_encode(trainer, seg_len, test_path, data_path, result_dir, flag='test'):
	
	files = sorted(glob.glob(os.path.join(test_path, '*.wav')))
	feeds = []

	for line in files:
		line = line.split('/')[-1]
		feeds.append({'s_id' : line.split('_')[0],
					  'utt_id' : line.split('_')[1].split('.')[0]})

	print('[Tester] - Number of files to encoded: ', len(feeds))
	dir_path = os.path.join(result_dir, f'{flag}/')
	os.makedirs(dir_path, exist_ok=True)

	with h5py.File(data_path, 'r') as f_h5:
		for feed in tqdm(feeds):

			src_speaker_spec = f_h5[f"test/{feed['s_id']}/{feed['utt_id']}/lin"][()]
			encode(src_speaker_spec, trainer, seg_len, s_speaker=feed['s_id'], utt_id=feed['utt_id'], result_dir=dir_path)
			


def target_classify(trainer, seg_len, synthesis_list, result_dir, flag='test'):
	dir_path = os.path.join(result_dir, f'{flag}/')
	with open(synthesis_list, 'r') as f:
		file = f.readlines()
	acc = []
	for line in file:
		# get wav path
		line = line.split('\n')[0].split(' ')
		utt_id = line[0].split('/')[1].split('_')[1]
		tar_speaker = line[1]
		wav_path = os.path.join(dir_path, f'{tar_speaker}_{utt_id}.wav')
		
		# get spectrogram
		_, spec = get_spectrograms(wav_path)
		
		# padding spec
		if len(spec) < seg_len:
			padding = np.zeros((seg_len - spec.shape[0], spec.shape[1]))
			spec = np.concatenate((spec, padding), axis=0)
		
		# classification
		logits = []
		for idx in range(0, len(spec), seg_len):
			if idx + (seg_len*2) > len(spec):
				spec_frag = spec[idx:-1]
			else:
				spec_frag = spec[idx:idx+seg_len]

			if len(spec_frag) >= seg_len:
				x = torch.from_numpy(np.expand_dims(spec_frag[:seg_len, :], axis=0)).type(torch.FloatTensor)
				logit = trainer.classify(x)
				logits.append(logit)
			elif idx == 0:
				raise RuntimeError('Please check if input is too short!')
		logits = np.concatenate(logits, axis=0)
		logits = np.sum(logits, axis = 0)
		
		if logits[0] >= logits[1]:
			clf_speaker = 'V001'
		elif logits[1] > logits[0]:
			clf_speaker = 'V002'
		
		if clf_speaker == tar_speaker:
			acc.append(1)
			#print('[info]: {} is classified to {}'.format(wav_path, clf_speaker), end = '')
		else:
			acc.append(0)
			#print('[Error]: {} is classified to {}'.format(wav_path, clf_speaker))
	print('Classification Acc: {:.3f}'.format(np.sum(acc)/len(acc)))


def encode_for_tacotron(trainer, wav_path, result_path='./data/metadata.csv'):
	wavs = sorted(glob.glob(os.path.join(wav_path, '*.wav')))
	print('[Tester] - Number of wav files to encoded: ', len(wavs))

	enc_outputs = []
	for wav_path in files:
		_, spec = get_spectrograms(wav_path)
		encodings = encode(src_speaker_spec, trainer, seg_len, s_speaker=feed['s_id'], utt_id=feed['utt_id'], save=False)
		enc_outputs.append(encodings)

	# build encodings to index mapping
	idx = 0
	multi2idx = {}
	for output in enc_outputs:
		for encodings in output:
			for encoding in encodings:
				if encoding not in multi2idx:
					multi2idx[str(encoding)] = idx
					idx += 1
	print(len(multi2idx))
	print(multi2idx[0])

	
