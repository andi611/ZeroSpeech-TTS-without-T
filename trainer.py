# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ trainer.py ]
#   Synopsis     [ training algorithms ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from hps.hps import hp
from torch import nn
from torch import optim
from torch.autograd import Variable
from model.model import Encoder, Decoder
from model.model import TargetClassifier
from model.model import SpeakerClassifier
from model.model import PatchDiscriminator
from model.model import Enhanced_Generator, Spectrogram_Patcher
from model.tacotron_integrate.tacotron import Tacotron, learning_rate_decay
from model.tacotron_integrate.loss import TacotronLoss
from utils import Logger, cc, to_var
from utils import grad_clip, reset_grad
from utils import calculate_gradients_penalty


class Trainer(object):
	def __init__(self, hps, data_loader, g_mode, enc_mode, log_dir='./log/'):
		self.hps = hps
		self.data_loader = data_loader
		self.model_kept = []
		self.max_keep = hps.max_to_keep
		self.logger = Logger(log_dir)
		self.g_mode = g_mode
		self.enc_mode = enc_mode
		if self.g_mode == 'naive': 
			self.sample_weights = torch.ones(hps.n_speakers)
		else:
			self.sample_weights = torch.cat((torch.zeros(hps.n_speakers-hps.n_target_speakers), \
								    		 torch.ones(hps.n_target_speakers)), dim=0)
			self.shift_c = to_var(torch.from_numpy(np.array([int(hps.n_speakers-hps.n_target_speakers) \
						   					 for _ in range(hps.batch_size)])), requires_grad=False)
		self.build_model()

	def build_model(self):
		hps = self.hps
		ns = self.hps.ns
		enc_mode = self.enc_mode
		seg_len = self.hps.seg_len
		enc_size = self.hps.enc_size
		emb_size = self.hps.emb_size
		betas = (0.5, 0.9)

		#---stage one---#
		self.Encoder = cc(Encoder(ns=ns, dp=hps.enc_dp, enc_size=enc_size, seg_len=seg_len, enc_mode=enc_mode))
		self.Decoder = cc(Decoder(ns=ns, c_in=enc_size, c_h=emb_size, c_a=hps.n_speakers, seg_len=seg_len))
		self.SpeakerClassifier = cc(SpeakerClassifier(ns=ns, c_in=enc_size * enc_size if enc_mode == 'binary' else \
													  (2*enc_size if enc_mode == 'multilabel_binary' else enc_size), \
													  c_h=emb_size, n_class=hps.n_speakers, dp=hps.dis_dp, seg_len=seg_len))
		
		#---stage one opts---#
		params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
		self.ae_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)
		self.clf_opt = optim.Adam(self.SpeakerClassifier.parameters(), lr=self.hps.lr, betas=betas)
		
		#---stage two---#
		if self.g_mode == 'naive':
			self.Generator = cc(Decoder(ns=ns, c_in=enc_size, c_h=emb_size, c_a=hps.n_speakers, seg_len=seg_len))
		elif self.g_mode == 'targeted' or self.g_mode == 'targeted_residual':
			self.Generator = cc(Decoder(ns=ns, c_in=enc_size, c_h=emb_size, c_a=hps.n_target_speakers, seg_len=seg_len, \
										output_mask=True if self.g_mode == 'targeted_residual' else False))
		elif self.g_mode == 'enhanced':
			self.Generator = cc(Enhanced_Generator(ns=ns, dp=hps.enc_dp, enc_size=1024, emb_size=1024, seg_len=seg_len, n_speakers=hps.n_speakers))
		elif self.g_mode == 'spectrogram':
			self.Generator = cc(Spectrogram_Patcher(ns=ns, c_in=513, c_h=emb_size, c_a=hps.n_target_speakers, seg_len=seg_len))
		elif self.g_mode == 'tacotron':
			self.Generator = cc(Tacotron(enc_size, hps.n_target_speakers, mel_dim=hp.n_mels, linear_dim=int(hp.n_fft/2)+1))
			self.tacotron_input_lengths = torch.tensor([self.hps.seg_len//8 for _ in range(hps.batch_size)])
		else:
			raise NotImplementedError('Invalid Generator mode!')
			
		self.PatchDiscriminator = cc(nn.DataParallel(PatchDiscriminator(ns=ns, n_class=hps.n_speakers \
																		if self.g_mode == 'naive' else hps.n_target_speakers,
																		seg_len=seg_len)))
		
		#---stage two opts---#
		self.gen_opt = optim.Adam(self.Generator.parameters(), lr=self.hps.lr, betas=betas)
		self.patch_opt = optim.Adam(self.PatchDiscriminator.parameters(), lr=self.hps.lr, betas=betas)
		
		
		#---target classifier---#
		self.TargetClassifier = cc(nn.DataParallel(TargetClassifier(ns=ns, n_class=2, seg_len=seg_len)))
		
		#---target classifier opts---#
		self.tclf_opt = optim.Adam(self.TargetClassifier.parameters(), lr=self.hps.lr, betas=betas)

	def reset_keep(self):
		self.model_kept = []

	def save_model(self, model_path, name, iteration, model_all=True):
		if model_all:
			all_model = {
				'encoder': self.Encoder.state_dict(),
				'decoder': self.Decoder.state_dict(),
				'generator': self.Generator.state_dict(),
				'classifier': self.SpeakerClassifier.state_dict(),
				'patch_discriminator': self.PatchDiscriminator.state_dict(),
				'target_classifier': self.TargetClassifier.state_dict(),
			}
		else:
			all_model = {
				'encoder': self.Encoder.state_dict(),
				'decoder': self.Decoder.state_dict(),
				'generator': self.Generator.state_dict(),
			}
		new_model_path = '{}-{}-{}'.format(model_path, name, iteration)
		torch.save(all_model, new_model_path)
		self.model_kept.append(new_model_path)

		if len(self.model_kept) >= self.max_keep:
			os.remove(self.model_kept[0])
			self.model_kept.pop(0)


	def load_model(self, model_path, load_model_list, verbose=True):
		if verbose: print('[Trainer] - load model from {}'.format(model_path))
		load_model_list = load_model_list.split(', ')
		all_model = torch.load(model_path)
		if verbose: print('[Trainer] - ', end = '')
		if 'encoder' in load_model_list:
			try:
				self.Encoder.load_state_dict(all_model['encoder'])
				if verbose: print('[encoder], ', end = '')
			except: print('[encoder - X], ', end = '')
		if 'decoder' in load_model_list:
			try:
				self.Decoder.load_state_dict(all_model['decoder'])
				if verbose: print('[decoder], ', end = '')
			except: print('[generator - X], ', end = '')
		if 'generator' in load_model_list:
			try:
				self.Generator.load_state_dict(all_model['generator'])
				if verbose: print('[generator], ', end = '')
			except: print('[generator - X], ', end = '')
		if 'classifier' in load_model_list:
			try:
				self.SpeakerClassifier.load_state_dict(all_model['classifier'])
				if verbose: print('[classifier], ', end = '')
			except: print('[classifier - X], ', end = '')
		if 'patch_discriminator' in load_model_list:
			try:
				self.PatchDiscriminator.load_state_dict(all_model['patch_discriminator'])
				if verbose: print('[patch_discriminator], ', end = '')
			except: print('[patch_discriminator - X], ', end = '')
		if 'target_classifier' in load_model_list:
			try:
				self.TargetClassifier.load_state_dict(all_model['target_classifier'])
				if verbose: print('[target_classifier], ', end = '')
			except: print('[target_classifier - X], ', end = '')
		if verbose: print('Loaded!')


	def add_duo_loader(self, source_loader, target_loader):
		self.source_loader = source_loader
		self.target_loader = target_loader


	def switch_loader(self, new_loader):
		self.data_loader = new_loader


	def set_eval(self):
		self.testing_shift_c = Variable(torch.from_numpy(np.array([int(self.hps.n_speakers-self.hps.n_target_speakers)]))).cuda()
		self.Encoder.eval()
		self.Decoder.eval()
		self.SpeakerClassifier.eval()
		self.PatchDiscriminator.eval()
		self.TargetClassifier.eval()
		if self.g_mode == 'tacotron': # keep dropout in Tacotron's decoder
			self.Generator.encoder.eval()
			self.Generator.postnet.eval()
		else:
			self.Generator.eval()


	def test_step(self, x, c, enc_only=False, verbose=True):
		self.set_eval()
		x = to_var(x).permute(0, 2, 1)
		enc, _ = self.Encoder(x)
		if enc_only or self.g_mode != 'tacotron': 
			x_dec = self.Decoder(enc, c)
		if not enc_only:
			if verbose: print('Testing with Autoencoder + Generator, encoding: ', enc.data.cpu().numpy())
			if self.g_mode != 'naive' and (c - self.testing_shift_c).data.cpu().numpy()[0] not in range(self.hps.n_target_speakers):
				raise RuntimeError('This generator can only convert to target speakers!')
			
			#---select Generator mode---#
			if self.g_mode == 'naive':
				x_dec += self.Generator(enc, c)
			elif self.g_mode == 'targeted':
				x_dec += self.Generator(enc, c - self.testing_shift_c)
			elif self.g_mode == 'targeted_residual':
				x_dec += (x_dec * self.Generator(enc, c - self.testing_shift_c)) / 2.0
			elif self.g_mode == 'enhanced' or self.g_mode == 'spectrogram':
				x_dec += self.Generator(x_dec, c - self.testing_shift_c)
			elif self.g_mode == 'tacotron':
				_, x_dec = self.Generator(enc, targets=None, speaker_id=(c - self.testing_shift_c), input_lengths=None)
			else:
				raise NotImplementedError('Invalid Generator mode!')
		else:
			if verbose: print('Testing with Autoencoder only, encoding: ', enc.data.cpu().numpy())
		
		return x_dec.data.cpu().numpy(), enc.data.cpu().numpy()


	def encoder_test_step(self, x):
		self.set_eval()
		x = to_var(x).permute(0, 2, 1)
		enc, _ = self.Encoder(x)
		return enc.data.cpu().numpy()

		
	def classify(self, x):
		self.set_eval()
		x = to_var(x).permute(0, 2, 1)
		logits = self.TargetClassifier(x)
		return logits.data.cpu().numpy()
		

	def permute_data(self, data, load_mel=False):
		C = to_var(data[0], requires_grad=False)
		X = to_var(data[1]).permute(0, 2, 1)
		if load_mel: 
			M = to_var(data[2]).permute(0, 2, 1)
			return C, X, M
		return C, X


	def sample_c(self, size):
		c_sample = Variable(torch.multinomial(self.sample_weights, 
							num_samples=size, replacement=True),  
							requires_grad=False)
		c_sample = c_sample.cuda() if torch.cuda.is_available() else c_sample
		return c_sample


	def encode_step(self, x):
		enc_act, enc = self.Encoder(x)
		return enc_act, enc


	def decode_step(self, enc, c):
		x_dec = self.Decoder(enc, c)
		return x_dec


	def patch_step(self, x, x_dec, is_dis=True):
		D_real, real_logits = self.PatchDiscriminator(x, classify=True)
		D_fake, fake_logits = self.PatchDiscriminator(x_dec, classify=True)
		if is_dis:
			w_dis = torch.mean(D_real - D_fake)
			gp = calculate_gradients_penalty(self.PatchDiscriminator, x, x_dec)
			return w_dis, real_logits, gp
		else:
			return -torch.mean(D_fake), fake_logits
	
	def tclf_step(self, x):
		logits = self.TargetClassifier(x)
		return logits
	

	def gen_step(self, enc, c):
		x_dec = self.Decoder(enc, c)
		if self.g_mode == 'naive':
			x_gen = x_dec + self.Generator(enc, c)
		elif self.g_mode == 'targeted':
			x_gen = x_dec + self.Generator(enc, c - self.shift_c)
		elif self.g_mode == 'targeted_residual':
			x_gen = (x_dec + (x_dec * self.Generator(enc, c - self.shift_c))) / 2.0
		elif self.g_mode == 'enhanced' or self.g_mode == 'spectrogram':
			x_gen = x_dec + self.Generator(x_dec, c - self.shift_c)
		else:
			raise NotImplementedError('Invalid generator mode to call gen_step()!')
		return x_gen 


	def clf_step(self, enc):
		logits = self.SpeakerClassifier(enc)
		return logits


	def tacotron_step(self, enc, m, c):
		m_dec, x_dec = self.Generator(enc, m, c - self.shift_c, input_lengths=self.tacotron_input_lengths)
		return m_dec, x_dec # mel, linear


	def cal_loss(self, logits, y_true, shift=False):
		# calculate loss 
		criterion = nn.CrossEntropyLoss()
		if shift and self.g_mode != 'naive': 
			loss = criterion(logits, y_true - self.shift_c)
		else: 
			loss = criterion(logits, y_true)
		return loss


	def cal_acc(self, logits, y_true, shift=False):
		_, ind = torch.max(logits, dim=1)
		if shift:
			acc = torch.sum((ind == y_true - self.shift_c).type(torch.FloatTensor)) / y_true.size(0)
		else:
			acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
		return acc


	def train(self, model_path, flag='train', mode='train', target_guided=False):
		# load hyperparams
		hps = self.hps

		if mode == 'pretrain_AE':
			for iteration in range(hps.enc_pretrain_iters):
				data = next(self.data_loader)
				c, x = self.permute_data(data)
				
				# encode
				enc_act, enc = self.encode_step(x)
				x_dec = self.decode_step(enc_act, c)
				loss_rec = torch.mean(torch.abs(x_dec - x))
				reset_grad([self.Encoder, self.Decoder])
				loss_rec.backward()
				grad_clip([self.Encoder, self.Decoder], hps.max_grad_norm)
				self.ae_opt.step()
				
				# tb info
				info = {
					f'{flag}/pre_loss_rec': loss_rec.item(),
				}
				slot_value = (iteration + 1, hps.enc_pretrain_iters) + tuple([value for value in info.values()])
				log = 'pre_AE:[%06d/%06d], loss_rec=%.3f'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 'ae', iteration + 1)
			print()

		elif mode == 'pretrain_C':
			for iteration in range(hps.dis_pretrain_iters):
				
				data = next(self.data_loader)
				c, x = self.permute_data(data)
				
				# encode
				enc_act, enc = self.encode_step(x)
				
				# classify speaker
				logits = self.clf_step(enc)
				loss_clf = self.cal_loss(logits, c)
				
				# update 
				reset_grad([self.SpeakerClassifier])
				loss_clf.backward()
				grad_clip([self.SpeakerClassifier], hps.max_grad_norm)
				self.clf_opt.step()
				
				# calculate acc
				acc = self.cal_acc(logits, c)
				info = {
					f'{flag}/pre_loss_clf': loss_clf.item(),
					f'{flag}/pre_acc': acc,
				}
				slot_value = (iteration + 1, hps.dis_pretrain_iters) + tuple([value for value in info.values()])
				log = 'pre_C:[%06d/%06d], loss_clf=%.2f, acc=%.2f'
				
				print(log % slot_value, end='\r')
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 'c', iteration + 1)
			print()

		elif mode == 'train':
			for iteration in range(hps.iters):
				
				# calculate current alpha
				if iteration < hps.lat_sched_iters:
					current_alpha = hps.alpha_enc * (iteration / hps.lat_sched_iters)
				else:
					current_alpha = hps.alpha_enc
				
				#==================train D==================#
				for step in range(hps.n_latent_steps):
					data = next(self.data_loader)
					c, x = self.permute_data(data)
					
					# encode
					enc_act, enc = self.encode_step(x)
					
					# classify speaker
					logits = self.clf_step(enc)
					loss_clf = self.cal_loss(logits, c)
					loss = hps.alpha_dis * loss_clf
					
					# update 
					reset_grad([self.SpeakerClassifier])
					loss.backward()
					grad_clip([self.SpeakerClassifier], hps.max_grad_norm)
					self.clf_opt.step()
					
					# calculate acc
					acc = self.cal_acc(logits, c)
					info = {
						f'{flag}/D_loss_clf': loss_clf.item(),
						f'{flag}/D_acc': acc,
					}
					slot_value = (step, iteration + 1, hps.iters) + tuple([value for value in info.values()])
					log = 'D-%d:[%06d/%06d], loss_clf=%.2f, acc=%.2f'
					
					print(log % slot_value, end='\r')
					if iteration % 100 == 0:
						for tag, value in info.items():
							self.logger.scalar_summary(tag, value, iteration + 1)
							
				#==================train G==================#
				data = next(self.data_loader)
				c, x = self.permute_data(data)
				
				# encode
				enc_act, enc = self.encode_step(x)
				
				# decode
				x_dec = self.decode_step(enc_act, c)
				loss_rec = torch.mean(torch.abs(x_dec - x))
				
				# classify speaker
				logits = self.clf_step(enc)
				acc = self.cal_acc(logits, c)
				loss_clf = self.cal_loss(logits, c)
				
				# maximize classification loss
				loss = loss_rec - current_alpha * loss_clf
				reset_grad([self.Encoder, self.Decoder])
				loss.backward()
				grad_clip([self.Encoder, self.Decoder], hps.max_grad_norm)
				self.ae_opt.step()
				
				info = {
					f'{flag}/loss_rec': loss_rec.item(),
					f'{flag}/G_loss_clf': loss_clf.item(),
					f'{flag}/alpha': current_alpha,
					f'{flag}/G_acc': acc,
				}
				slot_value = (iteration + 1, hps.iters) + tuple([value for value in info.values()])
				log = 'G:[%06d/%06d], loss_rec=%.3f, loss_clf=%.2f, alpha=%.2e, acc=%.2f'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 's1', iteration + 1)
			print()

		elif mode == 'patchGAN':
			for iteration in range(hps.patch_iters):
				#==================train D==================#
				for step in range(hps.n_patch_steps):
					
					data_s = next(self.source_loader)
					data_t = next(self.target_loader)
					_, x_s = self.permute_data(data_s)
					c, x_t = self.permute_data(data_t)
					
					# encode
					enc_act, _ = self.encode_step(x_s)
					
					# sample c
					c_prime = self.sample_c(x_t.size(0))
					
					# generator
					x_dec = self.gen_step(enc_act, c_prime)
					
					# discriminstor
					w_dis, real_logits, gp = self.patch_step(x_t, x_dec, is_dis=True)
					
					# aux classification loss 
					loss_clf = self.cal_loss(real_logits, c, shift=True)
					
					loss = -hps.beta_dis * w_dis + hps.beta_clf * loss_clf + hps.lambda_ * gp
					reset_grad([self.PatchDiscriminator])
					loss.backward()
					grad_clip([self.PatchDiscriminator], hps.max_grad_norm)
					self.patch_opt.step()
					
					# calculate acc
					acc = self.cal_acc(real_logits, c, shift=True)
					info = {
						f'{flag}/w_dis': w_dis.item(),
						f'{flag}/gp': gp.item(), 
						f'{flag}/real_loss_clf': loss_clf.item(),
						f'{flag}/real_acc': acc, 
					}
					slot_value = (step, iteration+1, hps.patch_iters) + tuple([value for value in info.values()])
					log = 'patch_D-%d:[%06d/%06d], w_dis=%.2f, gp=%.2f, loss_clf=%.2f, acc=%.2f'
					print(log % slot_value, end='\r')
					
					if iteration % 100 == 0:
						for tag, value in info.items():
							self.logger.scalar_summary(tag, value, iteration + 1)

				#==================train G==================#
				data_s = next(self.source_loader)
				data_t = next(self.target_loader)
				_, x_s = self.permute_data(data_s)
				c_t, x_t = self.permute_data(data_t)

				# encode
				enc_act, _ = self.encode_step(x_s)
				
				# sample c
				c_prime = self.sample_c(x_t.size(0))
				
				# generator
				x_dec = self.gen_step(enc_act, c_prime)
				
				# discriminstor
				loss_adv, fake_logits = self.patch_step(x_t, x_dec, is_dis=False)
				
				# aux classification loss 
				loss_clf = self.cal_loss(fake_logits, c_prime, shift=True)
				loss = hps.beta_clf * loss_clf + hps.beta_gen * loss_adv
				reset_grad([self.Generator])
				loss.backward()
				grad_clip([self.Generator], hps.max_grad_norm)
				self.gen_opt.step()

				if target_guided:
					# teacher forcing
					enc_tf, _ = self.encode_step(x_t)
					x_dec_tf = self.gen_step(enc_tf, c_t)
					loss_rec = torch.mean(torch.abs(x_dec_tf - x_t))
					reset_grad([self.Generator])
					loss_rec.backward()
					self.gen_opt.step()
				
				# calculate acc
				acc = self.cal_acc(fake_logits, c_prime, shift=True)
				info = {
					f'{flag}/loss_adv': loss_adv.item(),
					f'{flag}/fake_loss_clf': loss_clf.item(),
					f'{flag}/fake_acc': acc, 
					f'{flag}/tg_rec': loss_rec.item() if target_guided else 0.000, 
				}
				slot_value = (iteration+1, hps.patch_iters) + tuple([value for value in info.values()])
				log = 'patch_G:[%06d/%06d], loss_adv=%.2f, loss_clf=%.2f, acc=%.2f, tg_rec=%.3f'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 's2', iteration + 1)
			print()
		

		elif mode == 'autolocker':
			criterion = torch.nn.BCELoss()
			for iteration in range(hps.patch_iters):
				#==================train G==================#
				data_s = next(self.source_loader)
				data_t = next(self.target_loader)
				_, x_s = self.permute_data(data_s)
				c_t, x_t = self.permute_data(data_t)

				# encode
				enc_act, _ = self.encode_step(x_s)
				
				# sample c
				c_prime = self.sample_c(x_t.size(0))
				
				# decode
				residual_output = self.gen_step(enc_act, c_prime)
				
				# re-encode
				re_enc, _ = self.encode_step(residual_output)
				
				# re-encode loss
				loss_reenc = criterion(re_enc, enc_act.data)
				reset_grad([self.Generator])
				loss_reenc.backward()
				grad_clip([self.Generator], hps.max_grad_norm)
				self.gen_opt.step()

				if target_guided:
					# teacher forcing
					enc_tf, _ = self.encode_step(x_t)
					x_dec_tf = self.gen_step(enc_tf, c_t)
					loss_rec = torch.mean(torch.abs(x_dec_tf - x_t))
					reset_grad([self.Generator])
					loss_rec.backward()
					self.gen_opt.step()
				
				# calculate acc
				info = {
					f'{flag}/re_enc': loss_reenc.item(),
					f'{flag}/tg_rec': loss_rec.item() if target_guided else 0.000, 
				}
				slot_value = (iteration+1, hps.patch_iters) + tuple([value for value in info.values()])
				log = 'patch_G:[%06d/%06d], re_enc=%.3f, tg_rec=%.3f'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 's2', iteration + 1)
			print()
		
		elif mode == 't_classify':
			for iteration in range(hps.tclf_iters):
			#======train target classifier======#					
				data_t = next(self.target_loader)
				c, x_t = self.permute_data(data_t)
				
				# classification
				logits = self.tclf_step(x_t)
				
				# classification loss 
				loss = self.cal_loss(logits, c-self.shift_c)
				reset_grad([self.TargetClassifier])
				loss.backward()
				grad_clip([self.TargetClassifier], hps.max_grad_norm)
				self.tclf_opt.step()
				
				# calculate acc
				acc = self.cal_acc(logits, c-self.shift_c)
				info = {
					f'{flag}/acc': acc,
				}
				slot_value = (iteration+1, hps.tclf_iters) + tuple([value for value in info.values()])
				log = 'Target Classifier:[%05d/%05d], acc=%.2f'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 'tclf', iteration + 1)
			print()

		elif mode == 'train_Tacotron':
			
			assert self.g_mode == 'tacotron'
			criterion = TacotronLoss()
			self.Encoder.eval()

			for iteration in range(hps.tacotron_iters):
			#======train tacotron======#

				cur_lr = learning_rate_decay(init_lr=0.002, global_step=iteration)
				for param_group in self.gen_opt.param_groups:
					param_group['lr'] = cur_lr

				data = next(self.data_loader)
				c, x, m = self.permute_data(data, load_mel=True)
				
				# encode
				enc_act, enc = self.encode_step(x)

				# tacotron synthesis
				m_dec, x_dec = self.tacotron_step(enc_act.data, m, c)
				
				# reconstruction loss 
				loss_rec = criterion([m_dec, x_dec], [m, x])
				reset_grad([self.Generator])
				loss_rec.backward()
				grad_clip([self.Generator], hps.max_grad_norm)
				self.gen_opt.step()
				
				# tb info
				info = {
					f'{flag}/tacotron_loss_rec': loss_rec.item(),
					f'{flag}/tacotron_lr': cur_lr,
				}
				slot_value = (iteration + 1, hps.tacotron_iters) + tuple([value for value in info.values()])
				log = 'train_Tacotron:[%06d/%06d], loss_rec=%.3f, lr=%.2e'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 't', iteration + 1)
			print()

		else: 
			raise NotImplementedError()



