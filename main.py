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
from convert import test_from_list, cross_test, test_single, get_trainer
from dataloader import Dataset, DataLoader


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='zerospeech_project')
	parser.add_argument('--preprocess', default=False, action='store_true', help='preprocess the zerospeech dataset')
	parser.add_argument('--train', default=False, action='store_true', help='start stage 1 and stage 2 training')
	parser.add_argument('--test', default=False, action='store_true', help='test the trained model on the testing list provided at --synthesis_list')
	parser.add_argument('--cross_test', default=False, action='store_true', help='test the trained model on all testing files')
	parser.add_argument('--test_single', default=False, action='store_true', help='test the trained model on a single file')
	parser.add_argument('--load_model', default=False, action='store_true', help='whether to load training session from previous checkpoints')

	static_setting = parser.add_argument_group('static_setting')
	static_setting.add_argument('--flag', type=str, default='train', help='constant flag')
	static_setting.add_argument('--resample', type=bool, default=True, help='whether to just process the sampling procedure in the preprocessing process')
	static_setting.add_argument('--targeted_G', type=bool, default=True, help='G can only convert to target speakers and not all speakers')
	static_setting.add_argument('--one_hot', type=bool, default=bool(1), help='Set the encoder to encode to symbolic discrete one-hot vectors')
	static_setting.add_argument('--enc_only', type=bool, default=bool(1), help='whether to predict only with stage 1 audoencoder')
	static_setting.add_argument('--s_speaker', type=str, default='S015', help='for the --test_single mode, set voice convergence source speaker')
	static_setting.add_argument('--t_speaker', type=str, default='V001', help='for the --test_single mode, set voice convergence target speaker')
	
	data_path = parser.add_argument_group('data_path')
	data_path.add_argument('--source_path', type=str, default='./data/english/train/unit/', help='the zerospeech train unit dataset')
	data_path.add_argument('--target_path', type=str, default='./data/english/train/voice/', help='the zerospeech train voice dataset')
	data_path.add_argument('--test_path', type=str, default='./data/english/test/', help='the zerospeech test dataset')
	data_path.add_argument('--synthesis_list', type=str, default='./data/english/synthesis.txt/', help='the zerospeech testing list')
	data_path.add_argument('--dataset_path', type=str, default='./data/dataset.hdf5', help='the processed train dataset (unit + voice)')
	data_path.add_argument('--index_path', type=str, default='./data/index.json', help='sample training segments from the train dataset, for stage 1 training')
	data_path.add_argument('--index_source_path', type=str, default='./data/index_source.json', help='sample training source segments from the train dataset, for stage 2 training')
	data_path.add_argument('--index_target_path', type=str, default='./data/index_target.json', help='sample training target segments from the train dataset, for stage 2 training')
	data_path.add_argument('--speaker2id_path', type=str, default='./data/speaker2id.json', help='records speaker and speaker id')

	model_path = parser.add_argument_group('model_path')
	model_path.add_argument('--hps_path', type=str, default='./hps/zerospeech.json', help='hyperparameter path')
	model_path.add_argument('--ckpt_dir', type=str, default='./ckpt', help='checkpoint directory for training storage')
	model_path.add_argument('--result_dir', type=str, default='./result', help='result directory for generating test results')
	model_path.add_argument('--sub_result_dir', type=str, default='./english/', help='sub result directory for generating zerospeech synthesis results')
	model_path.add_argument('--model_name', type=str, default='model.pth', help='base model name for training')
	model_path.add_argument('--load_train_model_name', type=str, default='model.pth-ae-200000', help='the model to restore for training, the command --load_model will load this model')
	model_path.add_argument('--load_test_model_name', type=str, default='model.pth-s2-150000', help='the model to restore for testing, the command --test will load this model')
	args = parser.parse_args()

	HPS = Hps(args.hps_path)
	hps = HPS.get_tuple()

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
				   resample=args.resample)

	if args.train:
		
		#---create datasets---#
		dataset = Dataset(args.dataset_path, args.index_path, seg_len=hps.seg_len)
		sourceset = Dataset(args.dataset_path, args.index_source_path, seg_len=hps.seg_len)
		targetset = Dataset(args.dataset_path, args.index_target_path, seg_len=hps.seg_len)
		
		#---create data loaders---#
		data_loader = DataLoader(dataset, hps.batch_size)
		source_loader = DataLoader(sourceset, hps.batch_size)
		target_loader = DataLoader(targetset, hps.batch_size)
		
		#---handle paths---#
		os.makedirs(args.ckpt_dir, exist_ok=True)
		model_path = os.path.join(args.ckpt_dir, args.model_name)

		#---initialize trainer---#
		trainer = Trainer(hps, data_loader, args.targeted_G, args.one_hot)
		if args.load_model: trainer.load_model(os.path.join(args.ckpt_dir, args.load_train_model_name), model_all=False)

		if args.train:
			# trainer.train(model_path, args.flag, mode='pretrain_AE') # Stage 1 pre-train: encoder-decoder reconstruction
			trainer.train(model_path, args.flag, mode='pretrain_C')  # Stage 1 pre-train: classifier-1
			trainer.train(model_path, args.flag, mode='train') 		 # Stage 1 training
			
			# trainer.add_duo_loader(source_loader, target_loader)
			# trainer.train(model_path, args.flag, mode='patchGAN')	# Stage 2 training

	if args.test or args.test_single:

		os.makedirs(args.result_dir, exist_ok=True)
		model_path = os.path.join(args.ckpt_dir, args.load_test_model_name)
		trainer = get_trainer(args.hps_path, model_path, args.targeted_G, args.one_hot)

		if args.test:
			os.makedirs(os.path.join(args.result_dir, args.sub_result_dir), exist_ok=True)
			test_from_list(trainer, hps.seg_len, args.synthesis_list, args.dataset_path, args.speaker2id_path, args.result_dir, args.enc_only)
		if args.cross_test:
			cross_test(trainer, hps.seg_len, args.dataset_path, args.speaker2id_path, args.result_dir, args.enc_only, flag='test')
		if args.test_single:
			test_single(trainer, hps.seg_len, args.speaker2id_path, args.result_dir, args.enc_only, args.s_speaker, args.t_speaker)

