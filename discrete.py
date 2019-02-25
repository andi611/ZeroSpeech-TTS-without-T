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
            f'{flag}/disc_loss_rec': loss_rec.item(),
        }
        slot_value = (iteration + 1, hyper_params.enc_pretrain_iters) + \
            tuple([value for value in info.values()])
        log = 'train_discrete:[%06d/%06d], loss_rec=%.3f'
        print(log % slot_value, end='\r')

        if iteration % 100 == 0:
            for tag, value in info.items():
                trainer.logger.scalar_summary(tag, value, iteration + 1)
        if (iteration + 1) % 1000 == 0:
            trainer.save_model(model_path, 'dc', iteration + 1)
    print()
