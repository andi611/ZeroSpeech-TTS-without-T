from argparse import ArgumentParser
from os import path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import cuda, nn, optim

from model import Decoder
from trainer import Trainer
from utils import grad_clip, reset_grad


def clustering(inputs, n_clusters):
    class LookUp:
        def __init__(self, k_means):
            self.k_means = k_means

        def __call__(self, inputs):
            cls = [self.k_means.predict(np.expand_dims(i, 0)) for i in inputs]
            return np.array([self.k_means.cluster_centers_[c.squeeze()] for c in cls])
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(inputs)
    return k_means, LookUp(k_means)


def train_discrete_decoder(trainer, look_up, model_path, flag='train'):
    # trainer is already trained
    hyper_params = trainer.hps

    tDecoder = trainer.Decoder
    dDecoder = Decoder(ns=tDecoder.ns, c_in=tDecoder.emb_size,
                       c_h=tDecoder.emb_size, c_a=hyper_params.n_speakers)

    for iteration in range(hyper_params.enc_pretrain_iters):
        data = next(trainer.data_loader)
        c, x = trainer.permute_data(data)

        encoded = trainer.encode_step(x)
        x = look_up(x)
        x_tilde = trainer.decode_step(encoded, c)
        loss_rec = torch.mean(torch.abs(x_tilde - x))
        reset_grad([trainer.Encoder, trainer.Decoder])
        loss_rec.backward()
        grad_clip([trainer.Encoder, trainer.Decoder],
                  trainer.hps.max_grad_norm)
        trainer.ae_opt.step()

        # tb info
        info = {
            f'{flag}/pre_loss_rec': loss_rec.item(),
        }
        slot_value = (iteration + 1, hyper_params.enc_pretrain_iters) + \
            tuple([value for value in info.values()])
        log = 'pre_AE:[%06d/%06d], loss_rec=%.3f'
        print(log % slot_value, end='\r')

        if iteration % 100 == 0:
            for tag, value in info.items():
                trainer.logger.scalar_summary(tag, value, iteration + 1)
        if (iteration + 1) % 1000 == 0:
            trainer.save_model(model_path, 'ae', iteration + 1)
    print()


def train(trainer, model_path, flag='train', mode='train'):
    # load hyperparams
    hps = trainer.hps

    if mode == 'pretrain_AE':
        for iteration in range(hps.enc_pretrain_iters):
            data = next(trainer.data_loader)
            c, x = trainer.permute_data(data)

            # encode
            enc = trainer.encode_step(x)
            x_tilde = trainer.decode_step(enc, c)
            loss_rec = torch.mean(torch.abs(x_tilde - x))
            reset_grad([trainer.Encoder, trainer.Decoder])
            loss_rec.backward()
            grad_clip([trainer.Encoder, trainer.Decoder],
                      trainer.hps.max_grad_norm)
            trainer.ae_opt.step()

            # tb info
            info = {
                f'{flag}/pre_loss_rec': loss_rec.item(),
            }
            slot_value = (iteration + 1, hps.enc_pretrain_iters) + \
                tuple([value for value in info.values()])
            log = 'pre_AE:[%06d/%06d], loss_rec=%.3f'
            print(log % slot_value, end='\r')

            if iteration % 100 == 0:
                for tag, value in info.items():
                    trainer.logger.scalar_summary(tag, value, iteration + 1)
            if (iteration + 1) % 1000 == 0:
                trainer.save_model(model_path, 'ae', iteration + 1)
        print()

    elif mode == 'pretrain_C':
        for iteration in range(hps.dis_pretrain_iters):

            data = next(trainer.data_loader)
            c, x = trainer.permute_data(data)

            # encode
            enc = trainer.encode_step(x)

            # classify speaker
            logits = trainer.clf_step(enc)
            loss_clf = trainer.cal_loss(logits, c)

            # update
            reset_grad([trainer.SpeakerClassifier])
            loss_clf.backward()
            grad_clip([trainer.SpeakerClassifier], trainer.hps.max_grad_norm)
            trainer.clf_opt.step()

            # calculate acc
            acc = trainer.cal_acc(logits, c)
            info = {
                f'{flag}/pre_loss_clf': loss_clf.item(),
                f'{flag}/pre_acc': acc,
            }
            slot_value = (iteration + 1, hps.dis_pretrain_iters) + \
                tuple([value for value in info.values()])
            log = 'pre_C:[%06d/%06d], loss_clf=%.2f, acc=%.2f'

            print(log % slot_value, end='\r')
            if iteration % 100 == 0:
                for tag, value in info.items():
                    trainer.logger.scalar_summary(tag, value, iteration + 1)
            if (iteration + 1) % 1000 == 0:
                trainer.save_model(model_path, 'c', iteration + 1)
        print()

    elif mode == 'train':
        for iteration in range(hps.iters):

            # calculate current alpha
            if iteration < hps.lat_sched_iters:
                current_alpha = hps.alpha_enc * \
                    (iteration / hps.lat_sched_iters)
            else:
                current_alpha = hps.alpha_enc

            #==================train D==================#
            for step in range(hps.n_latent_steps):
                data = next(trainer.data_loader)
                c, x = trainer.permute_data(data)

                # encode
                enc = trainer.encode_step(x)

                # classify speaker
                logits = trainer.clf_step(enc)
                loss_clf = trainer.cal_loss(logits, c)
                loss = hps.alpha_dis * loss_clf

                # update
                reset_grad([trainer.SpeakerClassifier])
                loss.backward()
                grad_clip([trainer.SpeakerClassifier],
                          trainer.hps.max_grad_norm)
                trainer.clf_opt.step()

                # calculate acc
                acc = trainer.cal_acc(logits, c)
                info = {
                    f'{flag}/D_loss_clf': loss_clf.item(),
                    f'{flag}/D_acc': acc,
                }
                slot_value = (step, iteration + 1, hps.iters) + \
                    tuple([value for value in info.values()])
                log = 'D-%d:[%06d/%06d], loss_clf=%.2f, acc=%.2f'

                print(log % slot_value, end='\r')
                if iteration % 100 == 0:
                    for tag, value in info.items():
                        trainer.logger.scalar_summary(
                            tag, value, iteration + 1)
            #==================train G==================#
            data = next(trainer.data_loader)
            c, x = trainer.permute_data(data)

            # encode
            enc = trainer.encode_step(x)

            # decode
            x_tilde = trainer.decode_step(enc, c)
            loss_rec = torch.mean(torch.abs(x_tilde - x))

            # classify speaker
            logits = trainer.clf_step(enc)
            acc = trainer.cal_acc(logits, c)
            loss_clf = trainer.cal_loss(logits, c)

            # maximize classification loss
            loss = loss_rec - current_alpha * loss_clf
            reset_grad([trainer.Encoder, trainer.Decoder])
            loss.backward()
            grad_clip([trainer.Encoder, trainer.Decoder],
                      trainer.hps.max_grad_norm)
            trainer.ae_opt.step()

            info = {
                f'{flag}/loss_rec': loss_rec.item(),
                f'{flag}/G_loss_clf': loss_clf.item(),
                f'{flag}/alpha': current_alpha,
                f'{flag}/G_acc': acc,
            }
            slot_value = (iteration + 1, hps.iters) + \
                tuple([value for value in info.values()])
            log = 'G:[%06d/%06d], loss_rec=%.3f, loss_clf=%.2f, alpha=%.2e, acc=%.2f'
            print(log % slot_value, end='\r')

            if iteration % 100 == 0:
                for tag, value in info.items():
                    trainer.logger.scalar_summary(tag, value, iteration + 1)
            if (iteration + 1) % 1000 == 0:
                trainer.save_model(model_path, 's1', iteration + 1)
        print()

    elif mode == 'patchGAN':
        for iteration in range(hps.patch_iters):
            #==================train D==================#
            for step in range(hps.n_patch_steps):

                data_s = next(trainer.source_loader)
                data_t = next(trainer.target_loader)
                _, x_s = trainer.permute_data(data_s)
                c, x_t = trainer.permute_data(data_t)

                # encode
                enc = trainer.encode_step(x_s)

                # sample c
                c_prime = trainer.sample_c(x_t.size(0))

                # generator
                x_tilde = trainer.gen_step(enc, c_prime)

                # discriminstor
                w_dis, real_logits, gp = trainer.patch_step(
                    x_t, x_tilde, is_dis=True)

                # aux classification loss
                loss_clf = trainer.cal_loss(real_logits, c, shift=True)

                loss = -hps.beta_dis * w_dis + hps.beta_clf * loss_clf + hps.lambda_ * gp
                reset_grad([trainer.PatchDiscriminator])
                loss.backward()
                grad_clip([trainer.PatchDiscriminator],
                          trainer.hps.max_grad_norm)
                trainer.patch_opt.step()

                # calculate acc
                acc = trainer.cal_acc(real_logits, c, shift=True)
                info = {
                    f'{flag}/w_dis': w_dis.item(),
                    f'{flag}/gp': gp.item(),
                    f'{flag}/real_loss_clf': loss_clf.item(),
                    f'{flag}/real_acc': acc,
                }
                slot_value = (step, iteration+1, hps.patch_iters) + \
                    tuple([value for value in info.values()])
                log = 'patch_D-%d:[%06d/%06d], w_dis=%.2f, gp=%.2f, loss_clf=%.2f, acc=%.2f'
                print(log % slot_value, end='\r')

                if iteration % 100 == 0:
                    for tag, value in info.items():
                        trainer.logger.scalar_summary(
                            tag, value, iteration + 1)

            #==================train G==================#
            data_s = next(trainer.source_loader)
            data_t = next(trainer.target_loader)
            _, x_s = trainer.permute_data(data_s)
            c, x_t = trainer.permute_data(data_t)

            # encode
            enc = trainer.encode_step(x_s)

            # sample c
            c_prime = trainer.sample_c(x_t.size(0))

            # generator
            x_tilde = trainer.gen_step(enc, c_prime)

            # discriminstor
            loss_adv, fake_logits = trainer.patch_step(
                x_t, x_tilde, is_dis=False)

            # aux classification loss
            loss_clf = trainer.cal_loss(fake_logits, c_prime, shift=True)
            loss = hps.beta_clf * loss_clf + hps.beta_gen * loss_adv
            reset_grad([trainer.Generator])
            loss.backward()
            grad_clip([trainer.Generator], trainer.hps.max_grad_norm)
            trainer.gen_opt.step()

            # calculate acc
            acc = trainer.cal_acc(fake_logits, c_prime, shift=True)
            info = {
                f'{flag}/loss_adv': loss_adv.item(),
                f'{flag}/fake_loss_clf': loss_clf.item(),
                f'{flag}/fake_acc': acc,
            }
            slot_value = (iteration+1, hps.patch_iters) + \
                tuple([value for value in info.values()])
            log = 'patch_G:[%06d/%06d], loss_adv=%.2f, loss_clf=%.2f, acc=%.2f'
            print(log % slot_value, end='\r')

            if iteration % 100 == 0:
                for tag, value in info.items():
                    trainer.logger.scalar_summary(tag, value, iteration + 1)
            if (iteration + 1) % 1000 == 0:
                trainer.save_model(model_path, 's2',
                                   iteration + 1 + hps.iters)
        print()

    else:
        raise NotImplementedError()
