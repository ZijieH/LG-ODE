import torch.nn as nn
import lib.utils as utils


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        decoder = nn.Sequential(
           nn.Linear(latent_dim, input_dim))

        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


class Encoder(nn.Module):
    def __init__(self, output_dim, input_dim,hidden_dim,layer_num):
        super(Encoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        encoder = utils.create_net(input_dim,output_dim*2,layer_num,hidden_dim)

        utils.init_network_weights(encoder)
        self.encoder = encoder

    def forward(self, data):
        h = self.encoder(data)
        mu,std = self.split_mean_mu(h)
        return mu,std

    def split_mean_mu(self, h):
        last_dim = h.size()[-1] // 2
        mu,std = h[:, :last_dim], h[:, last_dim:]
        return mu,std




