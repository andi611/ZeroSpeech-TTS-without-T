# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ main.py ]
#   Synopsis     [ main function that runs everything with argument parser ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import argparse
from hps.hps import Hps
from trainer import Trainer
from preprocess import preprocess
from convert import test_from_list, cross_test, test_single, test_encode, target_classify, get_trainer, encode_for_tacotron
from dataloader import Dataset, DataLoader


###################
# ARGUMENT RUNNER #
###################
def argument_runner():
	parser = argparse.ArgumentParser(description='zerospeech_project')
	parser.add_argument('--preprocess', default=False, action='store_true', help='preprocess the zerospeech dataset')
	
	parser.add_argument('--train', default=False, action='store_true', help='start all training')
	parser.add_argument('--train_ae', default=False, action='store_true', help='start auto-encoder training')
	parser.add_argument('--train_p', default=False, action='store_true', help='start patcher-generator training')
	parser.add_argument('--train_tgat', default=False, action='store_true', help='start pathcer-generator training with teacher forcing')
	parser.add_argument('--train_al', default=False, action='store_true', help='start auto-locker training with teacher forcing')
	parser.add_argument('--train_c', default=False, action='store_true', help='start target classifier training')
	parser.add_argument('--train_t', default=False, action='store_true', help='start tacotron training')

	parser.add_argument('--test', default=False, action='store_true', help='test the trained model on the testing list provided at --synthesis_list')
	parser.add_argument('--test_asr', default=False, action='store_true', help='test the trained model with asr on the testing list provided at --synthesis_list')
	parser.add_argument('--cross_test', default=False, action='store_true', help='test the trained model on all testing files')
	parser.add_argument('--test_single', default=False, action='store_true', help='test the trained model on a single file')
	parser.add_argument('--test_encode', default=False, action='store_true', help='test the trained model encoding ability by generating encodings')
	parser.add_argument('--test_classify', default=False, action='store_true', help='classify speakers on all testing files')
	parser.add_argument('--encode', default=False, action='store_true', help='encode all wav files under --target_path')
	parser.add_argument('--load_model', default=False, action='store_true', help='whether to load training session from previous checkpoints')

	static_setting = parser.add_argument_group('static_setting')
	static_setting.add_argument('--flag', type=str, default='train', help='constant flag')
	static_setting.add_argument('--remake', type=bool, default=bool(0), help='whether to remake dataset.hdf5')
	static_setting.add_argument('--g_mode', choices=['naive', 'targeted', 'enhanced', 'spectrogram', 'tacotron', 'set_from_hps'], default='set_from_hps', help='different stage two generator settings')
	static_setting.add_argument('--enc_mode', choices=['continues', 'one_hot', 'binary', 'multilabel_binary', 'gumbel_t', 'set_from_hps'], default='set_from_hps', help='different output method for the encoder to generate encodings')
	static_setting.add_argument('--enc_only', default=False, action='store_true', help='whether to predict only with stage 1 audoencoder')
	static_setting.add_argument('--s_speaker', type=str, default='S015', help='for the --test_single mode, set voice convergence source speaker')
	static_setting.add_argument('--t_speaker', type=str, default='V002', help='for the --test_single mode, set voice convergence target speaker')
	static_setting.add_argument('--encode_t', choices=['V001', 'V002'], default=None, help='target to be encoded by --encode, must be specified (V001, or V002).')
	
	data_path = parser.add_argument_group('data_path')
	data_path.add_argument('--dataset', choices=['english', 'surprise'], default='english', help='which dataset to use')
	data_path.add_argument('--source_path', type=str, default='./data/english/train/unit/', help='the zerospeech train unit dataset')
	data_path.add_argument('--target_path', type=str, default='./data/english/train/voice/', help='the zerospeech train voice dataset')
	data_path.add_argument('--test_path', type=str, default='./data/english/test/', help='the zerospeech test dataset')
	data_path.add_argument('--synthesis_list', type=str, default='./data/english/synthesis.txt', help='the zerospeech testing list')
	data_path.add_argument('--dataset_path', type=str, default='./data/dataset_english.hdf5', help='the processed train dataset (unit + voice)')
	data_path.add_argument('--index_path', type=str, default='./data/index_english.json', help='sample training segments from the train dataset, for stage 1 training')
	data_path.add_argument('--index_source_path', type=str, default='./data/index_english_source.json', help='sample training source segments from the train dataset, for stage 2 training')
	data_path.add_argument('--index_target_path', type=str, default='./data/index_english_target.json', help='sample training target segments from the train dataset, for stage 2 training')
	data_path.add_argument('--speaker2id_path', type=str, default='./data/speaker2id_english.json', help='records speaker and speaker id')
	data_path.add_argument('--multi2idx_path', type=str, default='./data/multi2idx.json', help='records encoding and idx mapping')
	data_path.add_argument('--metadata_path', type=str, default='./data/metadata_english_target.csv', help='path to store encodings for Tacotron')

	model_path = parser.add_argument_group('model_path')
	model_path.add_argument('--hps_path', type=str, default='./hps/zerospeech_english.json', help='hyperparameter path, please refer to the default settings in zerospeech.json')
	model_path.add_argument('--ckpt_dir', type=str, default='./ckpt_english', help='checkpoint directory for training storage')
	model_path.add_argument('--result_dir', type=str, default='./result', help='result directory for generating test results')
	model_path.add_argument('--sub_result_dir', type=str, default='./english/', help='sub result directory for generating zerospeech synthesis results')
	model_path.add_argument('--model_name', type=str, default='model.pth', help='base model name for training')
	model_path.add_argument('--load_train_model_name', type=str, default='model.pth-ae-424000', help='the model to restore for training, the command --load_model will load this model')
	model_path.add_argument('--load_test_model_name', type=str, default='model.pth-s2-150000', help='the model to restore for testing, the command --test will load this model')
	args = parser.parse_args()
	
	#---reparse if switching dataset---#
	if args.dataset == 'surprise':
		for action in parser._actions:
			if ('path' in action.dest or 'synthesis_list' in action.dest or 'sub_result_dir' in action.dest or 'ckpt_dir' in action.dest):
				if 'english' in action.default:
					action.default = action.default.replace('english', 'surprise')
		args = parser.parse_args()
	print('[Runner] - Dataset: ', args.dataset)

	#---get hps---#
	HPS = Hps(args.hps_path)
	hps = HPS.get_tuple()
	
	#---show current mode---#
	if args.g_mode == 'set_from_hps': args.g_mode = hps.g_mode 
	if args.enc_mode == 'set_from_hps': args.enc_mode = hps.enc_mode 
	print('[Runner] - Generation mode: ', 'autoencoder only' if args.enc_only else 'with generator')
	print('[Runner] - Generator mode: ', args.g_mode)
	print('[Runner] - Encoder mode: ', args.enc_mode)
	print('[Runner] - Encoding dim: ', hps.enc_size)

	return args, hps


########
# MAIN #
########
def main():
	
	args, hps = argument_runner()

	if args.preprocess:	
		
		preprocess(args.source_path, 
				   args.target_path,
				   args.test_path,
				   args.dataset_path,  
				   args.index_path, 
				   args.index_source_path, 
				   args.index_target_path, 
				   args.speaker2id_path, 
				   seg_len=hps.seg_len,
				   n_samples=hps.n_samples,
				   dset=args.flag,
				   remake=args.remake)


	if args.train or args.train_ae or args.train_p or args.train_tgat or args.train_al or args.train_c or args.train_t:
		
		#---create datasets---#
		dataset = Dataset(args.dataset_path, args.index_path, seg_len=hps.seg_len)
		sourceset = Dataset(args.dataset_path, args.index_source_path, seg_len=hps.seg_len)
		targetset = Dataset(args.dataset_path, args.index_target_path, seg_len=hps.seg_len, load_mel=True if args.train_t else False)
		
		#---create data loaders---#
		data_loader = DataLoader(dataset, hps.batch_size)
		source_loader = DataLoader(sourceset, hps.batch_size)
		target_loader = DataLoader(targetset, hps.batch_size)
		
		#---handle paths---#
		os.makedirs(args.ckpt_dir, exist_ok=True)
		model_path = os.path.join(args.ckpt_dir, args.model_name)

		#---initialize trainer---#
		trainer = Trainer(hps, data_loader, args.g_mode, args.enc_mode)
		if args.load_model: trainer.load_model(os.path.join(args.ckpt_dir, args.load_train_model_name), load_model_list=hps.load_model_list)

		if args.train or args.train_ae:
			trainer.train(model_path, args.flag, mode='pretrain_AE') 	# Stage 1 pre-train: encoder-decoder reconstruction
			# trainer.train(model_path, args.flag, mode='pretrain_C')   # Deprecated: Stage 1 pre-train: classifier-1
			# trainer.train(model_path, args.flag, mode='train') 		# Deprecated: Stage 1 training
			trainer.reset_keep()

		if args.train or args.train_p or args.train_tgat:	
			trainer.add_duo_loader(source_loader, target_loader)
			trainer.train(model_path, args.flag, mode='patchGAN', target_guided=args.train_tgat)		# Stage 2 training
			trainer.reset_keep()

		if args.train or args.train_al:	
			trainer.add_duo_loader(source_loader, target_loader)
			trainer.train(model_path, args.flag, mode='autolocker', target_guided=True)		# Stage 2 training
			trainer.reset_keep()
			
		if args.train or args.train_c:	
			trainer.add_duo_loader(source_loader, target_loader)
			trainer.train(model_path, args.flag, mode='t_classify') 	# Target speaker classifier training
			trainer.reset_keep()

		if args.train or args.train_t:
			trainer.switch_loader(target_loader)
			trainer.train(model_path, args.flag, mode='train_Tacotron')
			trainer.reset_keep()


	if args.test or args.test_asr or args.cross_test or args.test_single or args.test_encode or args.test_classify or args.encode:

		os.makedirs(args.result_dir, exist_ok=True)
		model_path = os.path.join(args.ckpt_dir, args.load_test_model_name)
		trainer = get_trainer(args.hps_path, model_path, args.g_mode, args.enc_mode)

		if args.test or args.test_asr:
			result_dir = os.path.join(args.result_dir, args.sub_result_dir)
			os.makedirs(result_dir, exist_ok=True)
			test_from_list(trainer, hps.seg_len, args.synthesis_list, args.dataset_path, args.speaker2id_path, result_dir, args.enc_only, run_asr=args.test_asr)
		if args.cross_test:
			cross_test(trainer, hps.seg_len, args.dataset_path, args.speaker2id_path, args.result_dir, args.enc_only, flag='test')
		if args.test_single:
			test_single(trainer, hps.seg_len, args.speaker2id_path, args.result_dir, args.enc_only, args.s_speaker, args.t_speaker)
		if args.test_encode:
			result_dir = os.path.join(args.result_dir, args.sub_result_dir)
			os.makedirs(result_dir, exist_ok=True)
			test_encode(trainer, hps.seg_len, args.test_path, args.dataset_path, result_dir, flag='test')
		if args.test_classify:
			target_classify(trainer, hps.seg_len, args.synthesis_list, args.result_dir, flag='test')
		if args.encode:
			if args.encode_t == None:
				raise RuntimeError('Please specified encode target! (--encode_t=V001 or --encode_t=V002)')
			if hps.enc_size > 6:
				raise NotImplementedError('Not enough unique symbols to encode all the distinct units! See encode_for_tacotron() in convert.py')
			encode_for_tacotron(args.encode_t, trainer, hps.seg_len, args.multi2idx_path, wav_path=args.target_path, result_path=args.metadata_path)


if __name__ == '__main__':
	main()
