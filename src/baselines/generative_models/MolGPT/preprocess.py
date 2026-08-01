import os
import sys
import json
import pickle
import pandas as pd
from tqdm import tqdm
from collections import Counter
from src.baselines.sequences.transfer import SmilesToClearSmiles, SmilesToDSmiles, SmilesToSelfies
from src.baselines.sequences.transfer import SmilesToGroupSelfies, SmilesTotSmiles
from src.baselines.sequences.transfer import SmilesFromDSmiles, SmilesFromSelfies, SmilesFromGroupSelfies, SmilesFromtSmiles
from src.baselines.sequences.transfer import standardize_smiles
from src.fSMILES.fsmiles import SmilesToFSmiles, BatchSmilesFromFSmiles, SmilesFromFSmiles
from src.baselines.sequences.tokenizer.tokenize import tokenize_SMILES, tokenize_SMILES_SPE, tokenize_ClearSMILES
from src.baselines.sequences.tokenizer.tokenize import tokenize_deepSMILES, tokenize_SELFIES, tokenize_GroupSELFIES
from src.baselines.sequences.tokenizer.tokenize import tokenize_tSMILES, tokenize_fSMILES

def save_tokens(tokens, filename='selfies_tokens.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

def transform(smiles_list: list[str], seq: str='smiles'):
    if seq == 'smiles':
        toks = [tokenize_SMILES(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='smiles-tokenize')]
    elif seq == 'smiles-pair-encoding':
        toks = [tokenize_SMILES_SPE(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='smiles-pair-encoding-tokenize')]
    elif seq == 'clear-smiles':
        csmiles_list = [SmilesToClearSmiles(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='clear-smiles-transform')]
        toks = [tokenize_ClearSMILES(csmi) for csmi in tqdm(csmiles_list, file=sys.stdout, postfix='clear-smiles-tokenize')]
    elif seq == 'deep-smiles':
        dsmiles_list = [SmilesToDSmiles(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='deep-smiles-transform')]
        toks = [tokenize_deepSMILES(dsmi) for dsmi in tqdm(dsmiles_list, file=sys.stdout, postfix='deep-smiles-tokenize')]
    elif seq == 'selfies':
        selfies_list = [SmilesToSelfies(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='selfies-transform')]
        toks = [tokenize_SELFIES(sf) for sf in tqdm(selfies_list, file=sys.stdout, postfix='selfies-tokenize')]
    elif seq == 'group-selfies':
        gsf_list = [SmilesToGroupSelfies(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='group-selfies-transform')]
        toks = [tokenize_GroupSELFIES(gsf) for gsf in tqdm(gsf_list, file=sys.stdout, postfix='group-selfies-tokenize')]
    elif seq == 'tsmiles':
        tsmiles_list = [SmilesTotSmiles(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='tsmiles-transform')]
        toks = [tokenize_tSMILES(tsmi) for tsmi in tqdm(tsmiles_list, file=sys.stdout, postfix='tsmiles-tokenize')]
    elif seq == 'fsmiles-pair-encoding':
        fsmiles_list = [SmilesToFSmiles(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='fsmiles-transform')]
        toks = [tokenize_fSMILES(fsmi, unit_level=False) for fsmi in tqdm(fsmiles_list, file=sys.stdout, postfix='fsmiles-pair-encoding-tokenize')]
    elif seq == 'fsmiles':
        fsmiles_list = [SmilesToFSmiles(smi) for smi in tqdm(smiles_list, file=sys.stdout, postfix='fsmiles-branched-transform')]
        toks = [tokenize_fSMILES(fsmi, unit_level=True) for fsmi in tqdm(fsmiles_list, file=sys.stdout, postfix='fsmiles-tokenize')]
    else:
        raise Exception('Wrong type of sequence.')
    return toks

def inverse_transform(seq_list: list[str], seq: str='deep-smiles'):
    if seq == 'smiles':
        smiles_list = [standardize_smiles(smi) for smi in tqdm(seq_list, file=sys.stdout, postfix='smiles-transform')]
    elif seq =='smiles-pair-encoding':
        smiles_list = [standardize_smiles(smi) for smi in tqdm(seq_list, file=sys.stdout, postfix='smiles-pair-tokenize')]
    elif seq == 'clear-smiles':
        smiles_list = [standardize_smiles(smi) for smi in tqdm(seq_list, file=sys.stdout, postfix='clear-smiles-transform')]
    elif seq == 'deep-smiles':
        smiles_list = [SmilesFromDSmiles(s) for s in tqdm(seq_list, file=sys.stdout, postfix='deep-smiles-transform')]
    elif seq == 'selfies':
        smiles_list = [SmilesFromSelfies(s) for s in tqdm(seq_list, file=sys.stdout, postfix='selfies-transform')]
    elif seq == 'group-selfies':
        smiles_list = [SmilesFromGroupSelfies(s) for s in tqdm(seq_list, file=sys.stdout, postfix='group-selfies-transform')]
    elif seq == 'tsmiles':
        smiles_list = [SmilesFromtSmiles(s) for s in tqdm(seq_list, file=sys.stdout, postfix='tsmiles-transform')]
    elif seq == 'fsmiles-pair-encoding':
        smiles_list = BatchSmilesFromFSmiles(seq_list, check_fsmi=True)
    elif seq == 'fsmiles':
        smiles_list = BatchSmilesFromFSmiles(seq_list, check_fsmi=True)
    else:
        raise Exception('Wrong type of sequence.')
    return smiles_list

def parse(data_path: str, save_path: str, seq: str='smiles', max_len: int=None, sequence_save_path: str=None):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(data_path, 'r') as f:
        smis = f.read().splitlines()[1:]
    n_smis = len(smis)
    n_split = int(0.9 * n_smis)
    train_smis = smis[:n_split]
    valid_smis = smis[n_split:]
    train_props = [0] * len(train_smis)
    valid_props = [0] * len(valid_smis)

    toks = transform(smiles_list=smis, seq=seq)
    if max_len is None:
        lens = [len(tok) for tok in toks]
        lens_df = pd.DataFrame(lens, columns=['toks_len'])
        lens_df.to_csv(f'{save_path}/{seq}_token_lens.csv', index=False)

        count_lens_df = Counter(lens_df.toks_len)
        count_lens_df = dict(count_lens_df).items()
        count_lens_df = pd.DataFrame(count_lens_df, columns=['lens_df', 'count'])
        count_lens_df.sort_values('lens_df', ascending=False, inplace=True)
        count_lens_df.to_csv(f'{save_path}/{seq}_token_lens_count.csv', index=False)

        alphabet = []
        for tok in toks:
            alphabet.extend(tok)

        count_alphabet = Counter(alphabet)
        count_alphabet = dict(count_alphabet).items()
        count_alphabet = pd.DataFrame(count_alphabet, columns=['smiles', 'count'])
        count_alphabet.sort_values('count', ascending=False, inplace=True)
        count_alphabet.to_csv(f'{save_path}/{seq}_token_count.csv', index=False)

        alphabet.append(' ')
        alphabet = sorted(list(set(alphabet)))
        print('alphabet len:', len(alphabet))
        save_tokens(alphabet, filename=f'{save_path}/{seq}_token_alphabet.json')
        if sequence_save_path is not None:
            if not os.path.exists(sequence_save_path):
                os.makedirs(sequence_save_path)
            save_tokens(alphabet, filename=f'{sequence_save_path}/{seq}_token_alphabet.json')

        max_len = max(lens)
        print('max_len', max_len)

    train_toks = toks[:n_split]
    valid_toks = toks[n_split:]
    train_toks = [tok + [str(' ')] * (max_len - len(tok)) for tok in train_toks]
    valid_toks = [tok + [str(' ')] * (max_len - len(tok)) for tok in valid_toks]

    with open(f'{save_path}/train_toks.pkl', 'wb') as f:
        pickle.dump(train_toks, f)
    with open(f'{save_path}/valid_toks.pkl', 'wb') as f:
        pickle.dump(valid_toks, f)
