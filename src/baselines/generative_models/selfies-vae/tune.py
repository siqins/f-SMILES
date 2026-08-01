import os
import json
import yaml
import torch
from torch import nn
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import selfies as sf
from models import VAE_encode, VAE_decode, save_models
import warnings
warnings.filterwarnings('ignore')


def save_tokens(tokens, filename='selfies_tokens.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def load_tokens(filename='selfies_tokens.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess(train_path, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(train_path, 'r') as f:
        smis = f.read().splitlines()[1:]

    idx_traintest = int(len(smis) * 0.9)
    smis = smis[:idx_traintest]

    dataset = [sf.encoder(smi) for smi in smis]

    alphabet = sf.get_alphabet_from_selfies(dataset)
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet = list(sorted(alphabet))  # ['[=O]', '[C]', '[F]', '[O]', '[nop]']
    save_tokens(alphabet, filename=os.path.join(os.path.dirname(save_path), 'selfies_tokens.json'))

    pad_to_len = max(sf.len_selfies(s) for s in dataset)  # 5
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    mols = []
    for mol in dataset:
        label, one_hot = sf.selfies_to_encoding(
           selfies=mol,
           vocab_stoi=symbol_to_idx,
           pad_to_len=pad_to_len,
           enc_type="both"
        )
        mols.append(one_hot)

    one_hot = np.array(mols)
    print('SMILES', one_hot.shape)
    np.savez_compressed(save_path, arr=one_hot)


def train_model(data_train, data_valid, num_epochs, latent_dimension,
                lr_enc, lr_dec, KLD_alpha, checkpoint, sample_num, tensorBoard_graphing, model_encode, model_decode, device,
                num_batches_train, num_batches_valid, batch_size, save_path):
    optimizer_encoder = torch.optim.Adam(model_encode.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(model_decode.parameters(), lr=lr_dec)

    data_train=torch.tensor(data_train, dtype=torch.float)
    data_train=data_train.to(device)
    for epoch in range(num_epochs):
        x = [i for i in range(len(data_train))]
        random.shuffle(x)
        data_train = data_train[x]

        start = time.time()
        for batch_iteration in tqdm(range(num_batches_train)):
            loss, recon_loss, kld = 0., 0., 0.
            current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size
            inp_smile_hot = data_train[current_smiles_start : current_smiles_stop]
            inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2])
            latent_points, mus, log_vars = model_encode(inp_smile_encode)

            real_batch_size = latent_points.shape[0]
            latent_points = latent_points.reshape(1, real_batch_size, latent_points.shape[1])
            kld += -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
            hidden = model_decode.init_hidden(batch_size = real_batch_size)
            decoded_one_hot = torch.zeros(real_batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2]).to(device)
            for seq_index in range(inp_smile_hot.shape[1]):
                decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)
                decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
            decoded_one_hot = decoded_one_hot.reshape(real_batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2])
            _, label_atoms  = inp_smile_hot.max(2)
            label_atoms     = label_atoms.reshape(real_batch_size * inp_smile_hot.shape[1])
            criterion   = torch.nn.CrossEntropyLoss()
            recon_loss += criterion(decoded_one_hot, label_atoms)
            loss += recon_loss + KLD_alpha * kld
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model_decode.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 30 == 0:
                end = time.time()

                # assess reconstruction quality
                _, decoded_max_indices = decoded_one_hot.max(1)
                _, input_max_indices   = inp_smile_hot.reshape(real_batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2]).max(1)

                differences = 1. - torch.abs(decoded_max_indices - input_max_indices)
                differences = torch.clamp(differences, min = 0., max = 1.).double()
                quality     = 100. * torch.mean(differences)
                quality     = quality.detach().cpu().numpy()

                # qualityValid=quality_in_validation_set(data_valid)

                new_line = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| quality: %.4f )\tELAPSED TIME: %.5f' % (epoch, batch_iteration, num_batches_train, loss.item(), quality, end - start)
                print(new_line)
                start = time.time()

        save_models(model_encode, model_decode, epoch, save_path)


def train(
        data_path,
        save_path,

        epochs=100,
):
    data = np.load(data_path)['arr'].astype(np.float32)
    print('Data Acquired.')
    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_molec1Hot = len_max_molec * len_alphabet

    settings = yaml.load(open("original_code_from_paper/vae/settings.yml", "r"), Loader=yaml.FullLoader)
    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']
    training_parameters['num_epochs'] = epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_encode = VAE_encode(len_max_molec1Hot=len_max_molec1Hot, **encoder_parameter).to(device)
    model_decode = VAE_decode(len_alphabet=len_alphabet, **decoder_parameter).to(device)

    data = torch.tensor(data, dtype=torch.float).to(device)
    train_valid_test_size = [0.9, 0.1, 0.0]
    x = [i for i in range(len(data))]  # random shuffle input
    random.shuffle(x)
    data = data[x]
    idx_traintest = int(len(data) * train_valid_test_size[0])
    idx_trainvalid = idx_traintest + int(len(data) * train_valid_test_size[1])
    data_train = data[0:idx_traintest]
    data_valid = data[idx_traintest:idx_trainvalid]
    data_test = data[idx_trainvalid:]
    num_batches_train = int(len(data_train) / batch_size) + 1
    num_batches_valid = int(len(data_valid) / batch_size) + 1

    train_model(data_train=data_train, data_valid=data_valid, model_encode=model_encode,
                model_decode=model_decode, device=device, num_batches_train=num_batches_train,
                num_batches_valid=num_batches_valid, batch_size=batch_size, save_path=save_path,
                **training_parameters)


def sample(model_path, save_path, batch_size=250, sample_size=100, latent_dimension=25):

    len_max_molec = 126
    len_alphabet = 21
    len_max_molec1Hot = len_max_molec * len_alphabet

    alphabet = load_tokens('experiments/selfies_tokens.json')

    settings = yaml.load(open("original_code_from_paper/vae/settings.yml", "r"), Loader=yaml.FullLoader)
    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_encode = VAE_encode(len_max_molec1Hot=len_max_molec1Hot, **encoder_parameter)
    model_decode = VAE_decode(len_alphabet=len_alphabet, **decoder_parameter)
    model_encode.load_state_dict(torch.load(os.path.join(model_path, 'encoder_2.pth'), map_location=device))
    model_decode.load_state_dict(torch.load(os.path.join(model_path, 'decoder_2.pth'), map_location=device))
    model_encode = model_encode.to(device)
    model_decode = model_decode.to(device)
    model_encode.eval()
    model_decode.eval()


    smis = []
    for _ in tqdm(range(sample_size)):

        fancy_latent_point = torch.normal(torch.zeros(latent_dimension), torch.ones(latent_dimension))
        hidden = model_decode.init_hidden().to(device)
        gathered_atoms = []
        for ii in range(len_max_molec):                 # runs over letters from SMILES (len=size of largest molecule)
            fancy_latent_point = fancy_latent_point.reshape(1, 1, latent_dimension)
            fancy_latent_point=fancy_latent_point.to(device)
            decoded_one_hot, hidden = model_decode(fancy_latent_point, hidden)

            decoded_one_hot = decoded_one_hot.flatten()
            decoded_one_hot = decoded_one_hot.detach()

            soft = nn.Softmax(0)
            decoded_one_hot = soft(decoded_one_hot)
            _,MaxIdx=decoded_one_hot.max(0)
            gathered_atoms.append(MaxIdx.data.cpu().numpy().tolist())

        molecule=''
        for ii in gathered_atoms:
            molecule+=alphabet[ii]
        molecule=molecule.replace(' ','')
        smi = sf.decoder(molecule)
        smis.append(smi)

    df = pd.DataFrame(smis, columns=['smiles'])
    df.to_csv(os.path.join(save_path, 'samples_2.csv'), index=False)

if __name__ == '__main__':

    train_path = '../../../../datasets/fused_units_generated.csv'
    save_path = 'experiments/fused_units.npz'

    preprocess(train_path, save_path)

    train(data_path=save_path, save_path='experiments', epochs=200)

    model_path = 'experiments'
    save_path = 'experiments'
    start_time = time.time()
    sample(model_path=model_path, save_path=save_path, sample_size=10000)
    end_time = time.time()
    t = end_time - start_time
    print('spend time, ', t, 's')