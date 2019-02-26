from argparse import ArgumentParser
from os import path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import cuda, distributions, nn, optim
from torch.nn import functional as F

from dataloader import DataLoader, Dataset
from hps.hps import Hps
from model import Decoder
from trainer import Trainer
from utils import grad_clip, reset_grad


class GumbelSoftmax:
    def __init__(self, u=0, b=1, t=.1, dim=-1):
        self.gumbel = distributions.Gumbel(loc=u, scale=b)
        self.temperature = t
        self.dim = dim

    def __call__(self, inputs):
        sample = self.gumbel.sample(inputs.shape)
        x = torch.log(inputs)+sample
        return F.softmax(x/self.temperature, dim=self.dim)


class ClassEncoder:
    def __init__(self, input_dim, n_classes, hidden_dim=100):
        self.n_classes = n_classes
        self.linear_block = nn.ModuleList([
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=n_classes)
        ])
        self.input_dim = input_dim
        self.gumbel = GumbelSoftmax()

    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_dim)
        net = inputs
        for layer in self.linear_block:
            net = layer(net)
        return self.gumbel(net)


# class LatticeGumbel:
#     def __init__(self, n_lattices):
#         self.n_lattices = n_lattices
#         self.gumbel_softmax = GumbelSoftmax()

#     def __call__(self, inputs):
#         return None

#     def process(self, inputs, dim=-1):
#         normalized = self.normalize(inputs, dim=dim)
#         lattice = torch.floor(normalized * self.n_lattices).float()
#         lattice = torch
#         gumbel = self.gumbel_softmax(lattice)

#     @staticmethod
#     def normalize(inputs, dim=-1):
#         maximum = inputs.max(dim)[0]
#         minimum = inputs.min(dim)[0]
#         difference = maximum-minimum
#         return (inputs-minimum)/difference


def clustering(inputs, n_clusters):
    class LookUp:
        def __init__(self, k_means, shapes):
            self.k_means = k_means
            self.shapes = shapes

        def __call__(self, inputs, device='cuda'if cuda.is_available() else 'cpu'):
            inputs = np.array(inputs)
            if inputs.shape == self.shapes[1:]:
                inputs = np.expand_dims(inputs, 0)
            cls = np.array([self.k_means.predict(i.reshape(1, -1))
                            for i in inputs])
            return torch.tensor(np.array([self.k_means.cluster_centers_[c].reshape(*self.shapes[1:]) for c in cls]), device=device)
    k_means = KMeans(n_clusters=n_clusters)
    reshaped = inputs.reshape(inputs.shape[0], -1)
    k_means.fit(reshaped)
    return k_means, LookUp(k_means, shapes=inputs.shape)


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


def discrete_main(args):
    if not args.discrete:
        return
    HPS = Hps(args.hps_path)
    hps = HPS.get_tuple()
    model_path = path.join(args.ckpt_dir, args.load_test_model_name)
    dataset = Dataset(args.dataset_path, args.index_path, seg_len=hps.seg_len)
    data_loader = DataLoader(dataset, hps.batch_size)
    trainer = Trainer(hps, data_loader, args.targeted_G, args.one_hot)
    data = [d.unsqueeze(0) for d in dataset]
    data = [trainer.permute_data(d)[1] for d in data]
    encoded = [trainer.encode_step(x) for x in data]
    kmeans, look_up = clustering(encoded, n_clusters=args.n_clusters)
    train_discrete_decoder(trainer, look_up, model_path)
