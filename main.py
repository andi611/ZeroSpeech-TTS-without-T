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
from utils import Hps
from convert import test
from trainer import Trainer
from preprocess import preprocess
from dataloader import Dataset, DataLoader


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--preprocess', default=False, action='store_true')
	parser.add_argument('--train', default=False, action='store_true')
	parser.add_argument('--test', default=False, action='store_true')
	parser.add_argument('--load_model', default=False, action='store_true')

	parser.add_argument('--flag', type=str, default='train')
	parser.add_argument('--targeted_G', type=bool, default=bool(1))
	
	# mode_args = parser.add_argument_group('mode')
	parser.add_argument('--source_path', type=str, default='./data/english/train/unit/')
	parser.add_argument('--target_path', type=str, default='./data/english/train/voice/')
	parser.add_argument('--test_path', type=str, default='./data/english/test/')
	parser.add_argument('--dataset_path', type=str, default='./data/dataset.hdf5')
	parser.add_argument('--index_path', type=str, default='./data/index.json')
	parser.add_argument('--index_source_path', type=str, default='./data/index_source.json')
	parser.add_argument('--index_target_path', type=str, default='./data/index_target.json')
	parser.add_argument('--speaker2id_path', type=str, default='./data/speaker2id.json')

	parser.add_argument('--hps_path', type=str, default='./hps/zerospeech.json')
	parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
	parser.add_argument('--result_dir', type=str, default='./result')
	parser.add_argument('--model_name', type=str, default='model.pth')
	parser.add_argument('--load_train_model_name', type=str, default='model.pth-149999')
	parser.add_argument('--load_test_model_name', type=str, default='model.pth-149999')
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
				   dset=args.flag)

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
		trainer = Trainer(hps, data_loader, args.targeted_G)
		if args.load_model: trainer.load_model(args.load_train_model_name)

		if args.train:
			trainer.train(model_path, args.flag, mode='pretrain_R') # Stage 1 pre-train: encoder-decoder reconstruction
			trainer.train(model_path, args.flag, mode='pretrain_C') # Stage 1 pre-train: classifier-1
			trainer.train(model_path, args.flag, mode='train') 		# Stage 1 training
			
			# trainer.add_duo_loader(source_loader, target_loader)
			# trainer.train(model_path, args.flag, mode='patchGAN')	# Stage 2 training

	if args.test:

		os.makedirs(args.result_dir, exist_ok=True)
		model_path = os.path.join(args.ckpt_dir, args.load_model_name)
		test(args.dataset_path, model_path, args.hps_path, args.speaker2id_path, args.result_dir, args.targeted_G)


