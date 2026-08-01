import os
import re
import codecs
import time
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.learner import *
from SmilesPE.tokenizer import *
from sequences.tSMILES.t_smiles import seqs as ts_seqs
from fSMILES import fsmiles

def tokenize_SMILES(smiles: str):
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    toks = regex.findall(smiles.strip())
    return toks

def train_SPE(smiles_list: list):
    file_path = os.path.dirname(os.path.abspath(__file__))
    output = codecs.open(os.path.join(file_path, 'SPE.txt'), 'w')
    start = time.time()
    learn_SPE(smiles_list, output, 30000, min_frequency=2000, augmentation=1, verbose=True, total_symbols=True)
    end = time.time()
    print("Time elapsed:", end-start)

def tokenize_SMILES_SPE(smiles: str):
    file_path = os.path.dirname(os.path.abspath(__file__))
    spe_vob = codecs.open(os.path.join(file_path, 'SPE.txt'))
    spe = SPE_Tokenizer(spe_vob)
    toks = spe.tokenize(smiles)
    toks = toks.split(' ')
    return toks

def tokenize_ClearSMILES(cs: str):
    toks = atomwise_tokenizer(cs)
    return toks

def tokenize_deepSMILES(ds: str):
    toks = atomwise_tokenizer(ds)
    return toks

def tokenize_SELFIES(sf: str):
    toks = atomwise_tokenizer(sf)
    return toks

def tokenize_GroupSELFIES(gsf: str):
    toks = atomwise_tokenizer(gsf)
    return toks

def tokenize_tSMILES(ts: str):
    toks = ts_seqs.get_tokens_from_tsmiles(ts)
    return toks

def tokenize_fSMILES(fs: str, unit_level:bool=True):
    toks = fsmiles.FSmilesToTokens(fs, unit_level)
    return toks

if __name__ == '__main__':

    import pandas as pd
    from sequences.transfer import *

    # path = '../../data/fused_units_generated.csv'
    # smis = pd.read_csv(path)['smiles'].tolist()
    # train_SPE(smis)

    smi = 'C1(CC2=C3SC4=C2SC=C4)=C3C=C(CC5=C6SC7=C5SC=C7)C6=C1'

    print(tokenize_SMILES(smi))
    print()

    print(tokenize_SMILES_SPE(smi))
    print()

    cm = SmilesToClearSmiles(smi)
    print('ClearSmiles', cm)
    print(tokenize_ClearSMILES(cm))
    print()

    cm = SmilesToDSmiles(smi)
    print('DSmiles', cm)
    print(tokenize_deepSMILES(cm))
    print()

    cm = SmilesToSelfies(smi)
    print('Selfies', cm)
    print(tokenize_SELFIES(cm))
    print()

    cm = SmilesToGroupSelfies(smi)
    print('GroupSelfies', cm)
    print(tokenize_GroupSELFIES(cm))
    print()

    cm = SmilesTotSmiles(smi)
    print('tSmiles', cm)
    print(tokenize_tSMILES(cm))
    print()

    cm = fsmiles.SmileToFSmiles(smi)
    print('fSmiles', cm)
    print(tokenize_fSMILES(cm))