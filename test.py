import torch

from parallages import *

i = torch.randn(17, 513, 299)
x, z_mean, z_log_var = VariationalEncoder()(i)
kl_loss = (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).sum()*-.5
print(x.shape)
print(z_mean.shape)
print(z_log_var.shape)
print(kl_loss.shape)
