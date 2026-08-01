import os
import torch
from torch import nn



def _make_dir(directory):
    os.makedirs(directory)


def save_models(encoder, decoder, epoch, save_path):
    # out_dir = './saved_models/{}'.format(epoch)
    # _make_dir(out_dir)
    torch.save(encoder.state_dict(), os.path.join(save_path, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(save_path, 'decoder.pth'))



class VAE_encode(nn.Module):

    def __init__(self, layer_1d, layer_2d, layer_3d, latent_dimension, len_max_molec1Hot):
        """
        Fully Connected layers for the RNN.
        """
        super(VAE_encode, self).__init__()

        # Reduce dimension upto second last layer of Encoder
        self.encode_4d = nn.Sequential(
            nn.Linear(len_max_molec1Hot, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),

			nn.ReLU(),
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)


    def reparameterize(self, mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        #print('reparameterize(self, mu, log_var)')
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Go down to dim-4
        h1 = self.encode_4d(x)

        # Go down to dim-2 & produce the mean & variance
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var



class VAE_decode(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num, len_alphabet):
        """
        Through Decoder
        """
        super(VAE_decode, self).__init__()
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN  = nn.GRU(
                input_size  = latent_dimension,
                hidden_size = gru_neurons_num,
                num_layers  = gru_stack_size,
                batch_first = False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, len_alphabet),
        )


    def init_hidden(self, batch_size = 1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size, self.gru_neurons_num)


    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """
        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)        # fully connected layer

        return decoded, hidden


