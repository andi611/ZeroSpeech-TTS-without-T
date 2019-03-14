# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ eval_tacotron.py ]
#   Synopsis     [ Testing algorithms for a trained Tacotron model for the ZeroSpeech TTS-without-T project ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import json
import argparse
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
#--------------------------------#
import torch
from torch.autograd import Variable
#--------------------------------#
from model.tacotron import audio
from model.tacotron.text import text_to_sequence, symbols
from model.tacotron.tacotron import Tacotron
from convert import get_trainer, encode, parse_encodings
from preprocess import get_spectrograms
from hps.hps import hp, Hps


############
# CONSTANT #
############
USE_CUDA = torch.cuda.is_available()


#######################
# TEST CONFIGURATIONS #
#######################
def get_test_args():
	parser = argparse.ArgumentParser(description='testing arguments')
	parser.add_argument('--dataset', choices=['english', 'surprise'], default='english', help='which dataset are we testing')
	parser.add_argument('--eval_t', choices=['V001', 'V002', 'None'], default='None', help='target to be evalutated must be either (V001, or V002).')

	ckpt_parser = parser.add_argument_group('ckpt')
	ckpt_parser.add_argument('--ckpt_dir', type=str, default='./ckpt_tacotron_english/', help='path to the directory where model checkpoints are saved')
	ckpt_parser.add_argument('--model_name', type=str, default='checkpoint_step500000.pth-english-V002', help='name for the checkpoint file')
	ckpt_parser.add_argument('--encoder_path', type=str, default='./ckpt_english/model.pth-ae-400000-128-multi-6/', help='path to the encoder model')

	#---the arguments below will be handled automatically, should not change these---#
	path_parser = parser.add_argument_group('path')
	path_parser.add_argument('--result_dir', type=str, default='./result/', help='path to output test results')
	path_parser.add_argument('--sub_result_dir', type=str, default='./english/test', help='sub result directory for generating zerospeech synthesis results')
	path_parser.add_argument('--testing_dir', type=str, default='./data/english/test', help='path to the input test audios')
	path_parser.add_argument('--synthesis_list', type=str, default='./data/english/synthesis.txt', help='path to the input test transcripts')
	path_parser.add_argument('--speaker2id_path', type=str, default='./data/speaker2id_english.json', help='records speaker and speaker id')
	path_parser.add_argument('--multi2idx_path', type=str, default='./data/multi2idx_english_target.json', help='records encoding and idx mapping')
	path_parser.add_argument('--hps_path', type=str, default='./hps/zerospeech_english.json', help='hyperparameter path, please refer to the default settings in zerospeech.json')
	args = parser.parse_args()

	#---reparse if switching dataset---#
	if args.dataset == 'surprise':
		for action in parser._actions:
			if ('path' in action.dest or 'synthesis_list' in action.dest or 'dir' in action.dest):
				if 'english' in action.default:
					action.default = action.default.replace('english', 'surprise')
		args = parser.parse_args()

	return args

def valid_arguments(valid_target, arg):
	if not valid_target in arg:
		raise RuntimeWarning('The key word {} should be in the argument: {}, make sure you are running the correct file!'.format(valid_target, arg))


##################
# TEXT TO SPEECH #
##################
def tts(model, text):
	"""Convert text to speech waveform given a Tacotron model.
	"""
	if USE_CUDA:
		model = model.cuda()
	
	# NOTE: dropout in the decoder should be activated for generalization!
	# model.decoder.eval()
	model.encoder.eval()
	model.postnet.eval()

	sequence = np.array(text_to_sequence(text))
	sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
	if USE_CUDA:
		sequence = sequence.cuda()

	# Greedy decoding
	mel_outputs, linear_outputs, gate_outputs, alignments = model(sequence)

	linear_output = linear_outputs[0].cpu().data.numpy()
	spectrogram = audio._denormalize(linear_output)
	alignment = alignments[0].cpu().data.numpy()

	# Predicted audio signal
	waveform = audio.inv_spectrogram(linear_output.T)

	return waveform, alignment, spectrogram


####################
# SYNTHESIS SPEECH #
####################
def synthesis_speech(model, text, path):
	waveform, alignment, spectrogram = tts(model, text)
	librosa.output.write_wav(path, waveform, hp.sr)


########
# MAIN #
########
def main():

	#---initialize---#
	args = get_test_args()
	HPS = Hps(args.hps_path)
	hps = HPS.get_tuple()
	trainer = get_trainer(args.hps_path, args.encoder_path, hps.g_mode, hps.enc_mode)


	if args.eval_t == 'None':
		print('[Tacotron] - None is not a valid evaluation target! Please specify target manually, must be either V001, or V002.')
		return


	# Tacotron implementation: https://github.com/andi611/TTS-Tacotron-Pytorch
	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=256,
					 mel_dim=80,
					 linear_dim=1025,
					 r=5,
					 padding_idx=None,
					 attention='LocationSensitive',
					 use_mask=False)


	#---handle path---#
	result_dir = os.path.join(args.result_dir, args.sub_result_dir)
	os.makedirs(result_dir, exist_ok=True)
	checkpoint_path = os.path.join(args.ckpt_dir, args.model_name)
	if args.dataset == 'english' and not os.path.isdir('./ckpt_tacotron_english'):
		print('[Tacotron] - Recommand using the following name for ckpt_dir: ./ckpt_tacotron_english/')
	elif args.dataset == 'surprise' and not os.path.isdir('./ckpt_tacotron_surprise'):
		print('[Tacotron] - Recommand using the following name for ckpt_dir: ./ckpt_tacotron_surprise/')


	#---load and set model---#
	print('[Tacotron] - Testing on the {} set.'.format(args.dataset) )
	print('[Tacotron] - Loading model: ', checkpoint_path)
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint["state_dict"])


	#---load and set mappings---#
	print('[Tacotron] - Loading mapping files: ', args.speaker2id_path)
	valid_arguments(valid_target=args.dataset, arg=args.speaker2id_path)
	with open(args.speaker2id_path, 'r') as f_json:
		speaker2id = json.load(f_json)

	args.multi2idx_path = args.multi2idx_path.replace('target', args.eval_t)
	print('[Tacotron] - Loading mapping files: ', args.multi2idx_path)
	valid_arguments(valid_target=args.dataset, arg=args.multi2idx_path)
	valid_arguments(valid_target=args.eval_t, arg=args.multi2idx_path)
	with open(args.multi2idx_path, 'r') as f_json:
		multi2idx = json.load(f_json)


	#---parse testing list---#
	print('[Tacotron] - Testing from list: ', args.synthesis_list)
	valid_arguments(valid_target=args.dataset, arg=args.synthesis_list)
	feeds = []
	with open(args.synthesis_list, 'r') as f:
		file = f.readlines()
		for line in file:
			line = line.split('\n')[0].split(' ')
			feeds.append({'s_id' : line[0].split('/')[1].split('_')[0],
						  'utt_id' : line[0].split('/')[1].split('_')[1], 
						  't_id' : line[1], })
	print('[Tester] - Number of files to be resynthesize: ', len(feeds))

	for feed in tqdm(feeds):
		if feed['t_id'] == args.eval_t:
			wav_path = os.path.join(args.testing_dir, feed['s_id'] + '_' + feed['utt_id'] + '.wav')
			_, spec = get_spectrograms(wav_path)
			encodings = encode(spec, trainer, hps.seg_len, save=False)
			encodings = parse_encodings(encodings)
			line = ''.join([multi2idx[encoding] for encoding in encodings])
			print(line)
			out_path = os.path.join(result_dir, feed['t_id'] + '_' + feed['utt_id'] + '.wav')
			synthesis_speech(model, text=line, path=out_path)

	# model.decoder.max_decoder_steps = config.max_decoder_steps # Set large max_decoder steps to handle long sentence outputs
		

		
	sys.exit(0)

if __name__ == "__main__":
	main()
