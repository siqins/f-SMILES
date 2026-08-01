import numpy as np
import pandas as pd

CHARSET = [' ', '#', '(', ')', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
        'c', 'l', 'n', 'o', 'r', 's', 'e']

CHARSET = [' ', '#', '%', '(', ')', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']', 'c', 'l', 'n', 'o', 'r', 's', 'e']

class OneHotFeaturizer(object):
    def __init__(self, charset=CHARSET, padlength=120):
        self.charset = charset
        self.pad_length = padlength

    def featurize(self, smiles):
        feats = []
        for smi in smiles:
            f = self.one_hot_encode(smi)
            feats.append(f)
        feats = np.array(feats)
        return feats
        # return np.array([self.one_hot_encode(smi) for smi in smiles], dtype=object)

    def one_hot_array(self, i):
        return [int(x) for x in [ix == i for ix in range(len(self.charset))]]

    def one_hot_index(self, c):
        return self.charset.index(c)

    def pad_smi(self, smi):
        return smi.ljust(self.pad_length)

    def one_hot_encode(self, smi):
        return np.array([
            self.one_hot_array(self.one_hot_index(x)) for x in self.pad_smi(smi)
            ])

    def one_hot_decode(self, z):
        z1 = []
        for i in range(len(z)):
            s = ''
            for j in range(len(z[i])):
                oh = np.argmax(z[i][j])
                s += self.charset[oh]
            z1.append([s.strip()])
        return z1

    def decode_smiles_from_index(self, vec):
        return ''.join(map(lambda x: CHARSET[x], vec)).strip()

if __name__ == '__main__':

    path = r'../../../data/fused_units_clear.csv'
    path = r'../../../data/fused_units_generated.csv'
    smis = pd.read_csv(path)

    all_chars_len = []
    chars = []
    for smi in smis.smiles:
        s = list(smi)
        all_chars_len.append(len(s))
        chars.extend(s)

    print(max(all_chars_len))
    chars = list(set(chars)) + [' ']
    print(sorted(chars))

    print(len(chars))
    print(len(CHARSET))