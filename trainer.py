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
from torch import nn
from torch import optim
from torch.autograd import Variable
from model import Encoder, Decoder
from model import SpeakerClassifier
from model import PatchDiscriminator
from utils import Logger, cc, to_var
from utils import grad_clip, reset_grad
from utils import calculate_gradients_penalty


class Trainer(object):
	def __init__(self, hps, data_loader, targeted_G, one_hot, log_dir='./log/'):
		self.hps = hps
		self.data_loader = data_loader
		self.model_kept = []
		self.max_keep = hps.max_to_keep
		self.logger = Logger(log_dir)
		self.targeted_G = targeted_G
		self.one_hot = one_hot
		if not self.targeted_G: 
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
		emb_size = self.hps.emb_size
		betas = (0.5, 0.9)

		#---stage one---#
		self.Encoder = cc(Encoder(ns=ns, dp=hps.enc_dp, emb_size=emb_size, one_hot=self.one_hot))
		self.Decoder = cc(Decoder(ns=ns, c_in=emb_size, c_h=emb_size, c_a=hps.n_speakers, one_hot=self.one_hot))
		self.SpeakerClassifier = cc(SpeakerClassifier(ns=ns, c_in=emb_size, c_h=emb_size, n_class=hps.n_speakers, dp=hps.dis_dp, seg_len=hps.seg_len))
		
		#---stage one opts---#
		params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
		self.ae_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)
		self.clf_opt = optim.Adam(self.SpeakerClassifier.parameters(), lr=self.hps.lr, betas=betas)
		
		#---stage two---#
		self.Generator = cc(Decoder(ns=ns, c_in=emb_size, c_h=emb_size, c_a=hps.n_speakers if not self.targeted_G else hps.n_target_speakers, one_hot=self.one_hot))
		self.PatchDiscriminator = cc(nn.DataParallel(PatchDiscriminator(ns=ns, n_class=hps.n_speakers \
																		if not self.targeted_G else hps.n_target_speakers,
																		seg_len=hps.seg_len)))
		
		#---stage two opts---#
		self.gen_opt = optim.Adam(self.Generator.parameters(), lr=self.hps.lr, betas=betas)
		self.patch_opt = optim.Adam(self.PatchDiscriminator.parameters(), lr=self.hps.lr, betas=betas)


	def save_model(self, model_path, name, iteration, model_all=True):
		if model_all:
			all_model = {
				'encoder': self.Encoder.state_dict(),
				'decoder': self.Decoder.state_dict(),
				'generator': self.Generator.state_dict(),
				'classifier': self.SpeakerClassifier.state_dict(),
				'patch_discriminator': self.PatchDiscriminator.state_dict(),
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


	def load_model(self, model_path, model_all=True, verbose=True):
		if verbose: print('[Trainer] - load model from {}'.format(model_path))
		all_model = torch.load(model_path)
		self.Encoder.load_state_dict(all_model['encoder'])
		self.Decoder.load_state_dict(all_model['decoder'])
		self.Generator.load_state_dict(all_model['generator'])
		if model_all:
			self.SpeakerClassifier.load_state_dict(all_model['classifier'])
			self.PatchDiscriminator.load_state_dict(all_model['patch_discriminator'])


	def add_duo_loader(self, source_loader, target_loader):
		self.source_loader = source_loader
		self.target_loader = target_loader


	def set_eval(self):
		self.testing_shift_c = Variable(torch.from_numpy(np.array([int(self.hps.n_speakers-self.hps.n_target_speakers)]))).cuda()
		self.Encoder.eval()
		self.Decoder.eval()
		self.Generator.eval()
		self.SpeakerClassifier.eval()
		self.PatchDiscriminator.eval()


	def test_step(self, x, c, enc_only=False, verbose=True):
		self.set_eval()
		x = to_var(x).permute(0, 2, 1)
		enc, _ = self.Encoder(x)
		x_tilde = self.Decoder(enc, c)
		if not enc_only:
			if verbose: print('Testing with Autoencoder + Generator, encoding: ', enc.data.cpu().numpy())
			if self.targeted_G and (c - self.testing_shift_c).data.cpu().numpy()[0] not in range(self.hps.n_target_speakers):
				raise RuntimeError('This generator can only convert to target speakers!')
			x_tilde += self.Generator(enc, c) if not self.targeted_G else self.Generator(enc, c - self.testing_shift_c)
		else:
			if verbose: print('Testing with Autoencoder only, encoding: ', enc.data.cpu().numpy())
		return x_tilde.data.cpu().numpy(), enc.data.cpu().numpy()


	def permute_data(self, data):
		C = to_var(data[0], requires_grad=False)
		X = to_var(data[1]).permute(0, 2, 1)
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
		x_tilde = self.Decoder(enc, c)
		return x_tilde


	def patch_step(self, x, x_tilde, is_dis=True):
		D_real, real_logits = self.PatchDiscriminator(x, classify=True)
		D_fake, fake_logits = self.PatchDiscriminator(x_tilde, classify=True)
		if is_dis:
			w_dis = torch.mean(D_real - D_fake)
			gp = calculate_gradients_penalty(self.PatchDiscriminator, x, x_tilde)
			return w_dis, real_logits, gp
		else:
			return -torch.mean(D_fake), fake_logits


	def gen_step(self, enc, c):
		x_gen = self.Decoder(enc, c)
		x_gen += self.Generator(enc, c) if not self.targeted_G else self.Generator(enc, c - self.shift_c) 
		return x_gen 


	def clf_step(self, enc):
		logits = self.SpeakerClassifier(enc)
		return logits


	def cal_loss(self, logits, y_true, shift=False):
		# calculate loss 
		criterion = nn.CrossEntropyLoss()
		if shift and self.targeted_G: 
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


	def train(self, model_path, flag='train', mode='train'):
		# load hyperparams
		hps = self.hps

		if mode == 'pretrain_AE':
			for iteration in range(hps.enc_pretrain_iters):
				data = next(self.data_loader)
				c, x = self.permute_data(data)
				
				# encode
				enc_act, enc = self.encode_step(x)
				x_tilde = self.decode_step(enc_act, c)
				loss_rec = torch.mean(torch.abs(x_tilde - x))
				reset_grad([self.Encoder, self.Decoder])
				loss_rec.backward()
				grad_clip([self.Encoder, self.Decoder], self.hps.max_grad_norm)
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
				grad_clip([self.SpeakerClassifier], self.hps.max_grad_norm)
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
					grad_clip([self.SpeakerClassifier], self.hps.max_grad_norm)
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
				x_tilde = self.decode_step(enc_act, c)
				loss_rec = torch.mean(torch.abs(x_tilde - x))
				
				# classify speaker
				logits = self.clf_step(enc)
				acc = self.cal_acc(logits, c)
				loss_clf = self.cal_loss(logits, c)
				
				# maximize classification loss
				loss = loss_rec - current_alpha * loss_clf
				reset_grad([self.Encoder, self.Decoder])
				loss.backward()
				grad_clip([self.Encoder, self.Decoder], self.hps.max_grad_norm)
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
					enc_act, enc = self.encode_step(x_s)
					
					# sample c
					c_prime = self.sample_c(x_t.size(0))
					
					# generator
					x_tilde = self.gen_step(enc_act, c_prime)
					
					# discriminstor
					w_dis, real_logits, gp = self.patch_step(x_t, x_tilde, is_dis=True)
					
					# aux classification loss 
					loss_clf = self.cal_loss(real_logits, c, shift=True)
					
					loss = -hps.beta_dis * w_dis + hps.beta_clf * loss_clf + hps.lambda_ * gp
					reset_grad([self.PatchDiscriminator])
					loss.backward()
					grad_clip([self.PatchDiscriminator], self.hps.max_grad_norm)
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
				c, x_t = self.permute_data(data_t)

				# encode
				enc_act, enc = self.encode_step(x_s)
				
				# sample c
				c_prime = self.sample_c(x_t.size(0))
				
				# generator
				x_tilde = self.gen_step(enc_act, c_prime)
				
				# discriminstor
				loss_adv, fake_logits = self.patch_step(x_t, x_tilde, is_dis=False)
				
				# aux classification loss 
				loss_clf = self.cal_loss(fake_logits, c_prime, shift=True)
				loss = hps.beta_clf * loss_clf + hps.beta_gen * loss_adv
				reset_grad([self.Generator])
				loss.backward()
				grad_clip([self.Generator], self.hps.max_grad_norm)
				self.gen_opt.step()
				
				# calculate acc
				acc = self.cal_acc(fake_logits, c_prime, shift=True)
				info = {
					f'{flag}/loss_adv': loss_adv.item(),
					f'{flag}/fake_loss_clf': loss_clf.item(),
					f'{flag}/fake_acc': acc, 
				}
				slot_value = (iteration+1, hps.patch_iters) + tuple([value for value in info.values()])
				log = 'patch_G:[%06d/%06d], loss_adv=%.2f, loss_clf=%.2f, acc=%.2f'
				print(log % slot_value, end='\r')
				
				if iteration % 100 == 0:
					for tag, value in info.items():
						self.logger.scalar_summary(tag, value, iteration + 1)
				if (iteration + 1) % 1000 == 0:
					self.save_model(model_path, 's2', iteration + 1 + hps.iters)
			print()
		
		else: 
			raise NotImplementedError()



