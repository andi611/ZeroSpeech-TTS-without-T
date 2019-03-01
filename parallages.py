import torch
from torch import nn
from torch.nn import functional as F

from model import (RNN, append_emb, gumbel_softmax, linear, pad_layer,
                   pixel_shuffle_1d, upsample)


class VariationalDecoder(nn.Module):
    def __init__(self, c_in=512, c_out=513, c_h=512, c_a=8, ns=0.2, *args, **kwargs):
        super().__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in, 2*c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv5 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.dense1 = nn.Linear(c_h, c_h)
        self.dense2 = nn.Linear(c_h, c_h)
        self.dense3 = nn.Linear(c_h, c_h)
        self.dense4 = nn.Linear(c_h, c_h)
        self.RNN = nn.GRU(input_size=c_h, hidden_size=c_h//2,
                          num_layers=1, bidirectional=True)
        self.dense5 = nn.Linear(2*c_h + c_h, c_h)
        self.linear = nn.Linear(c_h, c_out)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)
        self.ins_norm5 = nn.InstanceNorm1d(c_h)
        # embedding layer
        self.emb1 = nn.Embedding(c_a, c_h)
        self.emb2 = nn.Embedding(c_a, c_h)
        self.emb3 = nn.Embedding(c_a, c_h)
        self.emb4 = nn.Embedding(c_a, c_h)
        self.emb5 = nn.Embedding(c_a, c_h)

    def conv_block(self, x, conv_layers, norm_layer, emb, res=True):
        # first layer
        x_add = x + emb.view(emb.size(0), emb.size(1), 1)
        out = pad_layer(x_add, conv_layers[0])
        out = F.leaky_relu(out, negative_slope=self.ns)
        # upsample by pixelshuffle
        out = pixel_shuffle_1d(out, upscale_factor=2)
        out = out + emb.view(emb.size(0), emb.size(1), 1)
        out = pad_layer(out, conv_layers[1])
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            x_up = upsample(x, scale_factor=2)
            out = out + x_up
        return out

    def dense_block(self, x, emb, layers, norm_layer, res=True):
        out = x
        for layer in layers:
            out = out + emb.view(emb.size(0), emb.size(1), 1)
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x, c):
        # conv layer
        x, z, log_var = x
        x = x*(torch.exp(.5*log_var)) + z

        out = self.conv_block(
            x, [self.conv1, self.conv2], self.ins_norm1, self.emb1(c), res=True)
        out = self.conv_block(
            out, [self.conv3, self.conv4], self.ins_norm2, self.emb2(c), res=True)
        out = self.conv_block(
            out, [self.conv5, self.conv6], self.ins_norm3, self.emb3(c), res=True)
        # dense layer
        out = self.dense_block(out, self.emb4(
            c), [self.dense1, self.dense2], self.ins_norm4, res=True)
        out = self.dense_block(out, self.emb4(
            c), [self.dense3, self.dense4], self.ins_norm5, res=True)
        emb = self.emb5(c)
        out_add = out + emb.view(emb.size(0), emb.size(1), 1)
        # rnn layer
        out_rnn = RNN(out_add, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = append_emb(self.emb5(c), out.size(2), out)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)
        #out = torch.tanh(out)
        return out


class VariationalEncoder(nn.Module):
    def __init__(self, c_in=513, c_h1=128, c_h2=512, c_h3=128, ns=0.2, dp=0.5, emb_size=512, one_hot=False, *args, **kwargs):
        super().__init__()
        self.ns = ns
        self.one_hot = one_hot
        self.conv1s = nn.ModuleList(
            [nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 8)]
        )
        self.conv2 = nn.Conv1d(len(self.conv1s)*c_h1 +
                               c_in, c_h2, kernel_size=1)
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv7 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(c_h2, c_h2)
        self.dense2 = nn.Linear(c_h2, c_h2)
        self.dense3 = nn.Linear(c_h2, c_h2)
        self.dense4 = nn.Linear(c_h2, c_h2)
        self.RNN = nn.GRU(input_size=c_h2, hidden_size=c_h3,
                          num_layers=1, bidirectional=True)
        self.linear = nn.Linear(c_h2 + 2*c_h3, emb_size)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h2)
        self.ins_norm2 = nn.InstanceNorm1d(c_h2)
        self.ins_norm3 = nn.InstanceNorm1d(c_h2)
        self.ins_norm4 = nn.InstanceNorm1d(c_h2)
        self.ins_norm5 = nn.InstanceNorm1d(c_h2)
        self.ins_norm6 = nn.InstanceNorm1d(c_h2)
        # dropout layer
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.drop5 = nn.Dropout(p=dp)
        self.drop6 = nn.Dropout(p=dp)
        # mean and log_var
        self.mean = nn.GRU(input_size=c_h2, hidden_size=1)
        self.log_var = nn.GRU(input_size=c_h2, hidden_size=1)

    def conv_block(self, x, conv_layers, norm_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            x_pad = F.pad(x, pad=(0, x.size(2) % 2), mode='reflect')
            x_down = F.avg_pool1d(x_pad, kernel_size=2)
            out = x_down + out
        return out

    def dense_block(self, x, layers, norm_layers, res=True):
        out = x
        for layer in layers:
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = self.conv_block(out, [self.conv2], [
                              self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv3, self.conv4], [
                              self.ins_norm2, self.drop2])
        out = self.conv_block(out, [self.conv5, self.conv6], [
                              self.ins_norm3, self.drop3])
        out = self.conv_block(out, [self.conv7, self.conv8], [
                              self.ins_norm4, self.drop4])
        # dense layer
        out = self.dense_block(out, [self.dense1, self.dense2], [
                               self.ins_norm5, self.drop5], res=True)
        out = self.dense_block(out, [self.dense3, self.dense4], [
                               self.ins_norm6, self.drop6], res=True)
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.linear)

        mean = RNN(out, self.mean)
        log_var = RNN(out, self.log_var)

        if self.one_hot:
            out = gumbel_softmax(out)
        else:
            out = F.leaky_relu(out, negative_slope=self.ns)
        return out, mean, log_var


class VariationalSpeakerClassifier(nn.Module):
    def __init__(self, c_in=512, c_h=512, n_class=8, dp=0.1, ns=0.01, seg_len=128):
        super(SpeakerClassifier, self).__init__()
        self.dp, self.ns = dp, ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h//2, kernel_size=3)
        self.conv8 = nn.Conv1d(c_h//2, c_h//4, kernel_size=3)
        if seg_len == 128:
            self.conv9 = nn.Conv1d(c_h//4, n_class, kernel_size=16)
        elif seg_len == 64:
            self.conv9 = nn.Conv1d(c_h//4, n_class, kernel_size=8)
        else:
            raise NotImplementedError(
                'Segement length {} is not supported!'.format(seg_len))
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h//4)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        x, z, log_var = x
        x = x*(torch.exp(.5*log_var)) + z

        out = self.conv_block(x, [self.conv1, self.conv2], [
                              self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv3, self.conv4], [
                              self.ins_norm2, self.drop2], res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], [
                              self.ins_norm3, self.drop3], res=True)
        out = self.conv_block(out, [self.conv7, self.conv8], [
                              self.ins_norm4, self.drop4], res=False)
        out = self.conv9(out)
        out = out.view(out.size()[0], -1)
        return out
