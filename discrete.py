import json
import os
from argparse import ArgumentParser
from os import path

import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import cuda, distributions, nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from convert import convert_all_sp, test
from dataloader import DataLoader, Dataset
from hps.hps import Hps
from parallages import VariationalDecoder as Decoder
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


class ClassEncoder(nn.Module):
    def __init__(self, input_shape, n_classes, hidden_dim=100):
        super().__init__()
        self.input_shape = np.array(input_shape)
        self.linear_block = nn.ModuleList([
            nn.Linear(in_features=np.prod(
                input_shape[1:]), out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=n_classes)
        ])
        self.gumbel = GumbelSoftmax()

    def forward(self, inputs):
        inputs = inputs.view(self.input_shape[0], -1)
        net = inputs
        for layer in self.linear_block:
            net = layer(net)
        return self.gumbel(net)


class ClassDecoder(nn.Module):
    def __init__(self, original_shape, n_classes, hidden_dim=100):
        super().__init__()
        self.original_shape = np.array(original_shape)
        self.linear_block = nn.ModuleList([
            nn.Linear(in_features=n_classes, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=np.prod(original_shape[1:]))
        ])

    def forward(self, inputs):
        net = inputs
        for layer in self.linear_block:
            net = layer(net)
        return net.view(*self.original_shape)


class ToOneHot(nn.Module):
    def __init__(self, input_shape, n_classes, hidden_dim=100):
        super().__init__()
        self.enc = ClassEncoder(input_shape, n_classes, hidden_dim)
        self.dec = ClassDecoder(input_shape, n_classes, hidden_dim)

    def forward(self, inputs):
        encoded = self.enc(inputs)
        return encoded, self.dec(encoded)


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


def finetune_discrete_decoder(trainer, look_up, model_path, flag='train'):
    # trainer is already trained
    hyper_params = trainer.hps

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


def discrete_test(trainer, data_path, speaker2id_path, result_dir, enc_only, flag):

    f_h5 = h5py.File(data_path, 'r')

    print('[Tester] - Testing on the {}ing set...'.format(flag))
    if flag == 'test':
        source_speakers = sorted(list(f_h5['test'].keys()))
    elif flag == 'train':
        source_speakers = [s for s in sorted(
            list(f_h5['train'].keys())) if s[0] == 'S']
    target_speakers = [s for s in sorted(
        list(f_h5['train'].keys())) if s[0] == 'V']
    print('[Tester] - Source speakers: %i, Target speakers: %i' %
          (len(source_speakers), len(target_speakers)))

    with open(speaker2id_path, 'r') as f_json:
        speaker2id = json.load(f_json)

    print('[Tester] - Converting all testing utterances from source speakers to target speakers, this may take a while...')
    for speaker_S in tqdm(source_speakers):
        for speaker_T in target_speakers:
            assert speaker_S != speaker_T
            dir_path = os.path.join(result_dir, f'p{speaker_S}_p{speaker_T}')
            os.makedirs(dir_path, exist_ok=True)

            convert_all_sp(trainer,
                           data_path,
                           speaker_S,
                           speaker_T,
                           enc_only=enc_only,
                           dset=flag,
                           speaker2id=speaker2id,
                           result_dir=dir_path)


def discrete_main(args):
    if not args.discrete:
        return
    HPS = Hps(args.hps_path)
    hps = HPS.get_tuple()
    model_path = path.join(args.ckpt_dir, args.load_test_model_name)
    dataset = Dataset(args.dataset_path, args.index_path, seg_len=hps.seg_len)
    data_loader = DataLoader(dataset, hps.batch_size)
    trainer = Trainer(hps, data_loader, args.targeted_G, args.one_hot)
    trainer.load_model(
        path.join(args.ckpt_dir, args.load_train_model_name), model_all=True)
    data = [d.unsqueeze(0) for d in dataset]
    data = [trainer.permute_data(d)[1] for d in data]
    encoded = [trainer.encode_step(x) for x in data]
    kmeans, look_up = clustering(encoded, n_clusters=args.n_clusters)
    test(trainer, args.dataset_path, args.speaker2id_path,
         args.result_dir, args.enc_only, args.flag)
    finetune_discrete_decoder(trainer, look_up, model_path)
    discrete_test(trainer, args.dataset_path, args.speaker2id_path,
                  'discrete_'+args.result_dir, args.enc_only, args.flag)
