import os
import json
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize
from ClearSMILES.seqs import get_ClearSMILES, SMILES_REGEX
from DeepSMILES.seqs import DSMILES_converter, DecodeError
from Selfies import selfies as sf
from GroupSelfies import group_selfies as gsf
from tSMILES.t_smiles import seqs

file_path = os.path.realpath( __file__ )

def save_tokens(tokens, filename='selfies_tokens.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

def load_tokens(filename='selfies_tokens.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def standardize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)

        standardizer = MolStandardize.standardize.Standardizer()
        mol = standardizer.standardize(mol)

        uncharger = MolStandardize.charge.Uncharger()
        mol = uncharger.uncharge(mol)

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canonical_smiles

    except Exception as e:
        print(f"SMILES standardization fails: {smiles}, error: {e}")
        return None


def SmilesToClearSmiles(smiles: str) -> str:
    clear_smiles = get_ClearSMILES(smiles, nb_random=1_00_000, SMILES_REGEX=SMILES_REGEX)
    return clear_smiles

def SmilesFromClearSmiles(clear_smiles: str) -> str:
    return standardize_smiles(clear_smiles)


def SmilesToDSmiles(smiles: str) -> str:
    d_smiles = DSMILES_converter.encode(smiles)
    return d_smiles

def SmilesFromDSmiles(d_smiles: str) -> str:
    try:
        smiles = DSMILES_converter.decode(d_smiles)
        smiles = standardize_smiles(smiles)
    except DecodeError as e:
        smiles = None
        print("DecodeError! Error message was '%s'" % e.message)
    return smiles


def SmilesToSelfies(smiles: str) -> str:
    return sf.encoder(smiles)

def SmilesFromSelfies(selfies_string: str) -> str:
    smiles = sf.decoder(selfies_string)
    smiles = standardize_smiles(smiles)
    return smiles

def get_alphabet_from_selfies(smiles: list):
    alphabet = sf.get_alphabet_from_selfies(smiles)
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet = list(sorted(alphabet))
    return alphabet


def SmilesToGroupSelfies(smiles: str, grammar=None) -> str:
    if grammar is None:
        grammar = gsf.GroupGrammar.from_file(os.path.join(os.path.dirname(file_path), 'GroupSelfies/grammar_fragment.txt'))

    mol = Chem.MolFromSmiles(smiles)
    group_selfies = grammar.full_encoder(mol)
    return group_selfies

def SmilesFromGroupSelfies(group_selfies: str, grammar=None) -> str:
    if grammar is None:
        grammar = gsf.GroupGrammar.from_file(os.path.join(os.path.dirname(file_path), 'GroupSelfies/grammar_fragment.txt'))
    try:
        mol = grammar.decoder(group_selfies)
        smiles = Chem.MolToSmiles(mol)
        return standardize_smiles(smiles)
    except:
        return None

def get_alphabet_from_group_selfies(smiles: list):
    fragments = gsf.fragment_mols(smiles, convert=True, method='fraggle')
    vocab_fragment = dict([(f'frag{idx}', gsf.Group(f'frag{idx}', frag)) for idx, frag in enumerate(fragments)])
    grammar_fragment = gsf.GroupGrammar(vocab=vocab_fragment)

    alphabet = sf.get_alphabet_from_selfies(smiles)
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet = list(sorted(alphabet))
    return grammar_fragment, alphabet

def save_alphabet_from_group_selfies(grammar_fragment, alphabet, save_path) -> None:
    grammar_fragment.to_file(os.path.join(save_path, f'group_selfies_grammar.txt'))
    save_tokens(alphabet, os.path.join(save_path, f'group_selfies_tokens.json'))


def SmilesTotSmiles(smiles: str) -> str:
    tsmiles = seqs.to_tsmiles(smiles)
    return tsmiles

def SmilesFromtSmiles(tsmiles: str) -> str:
    smiles = seqs.from_tsmiles(tsmiles)
    smiles = standardize_smiles(smiles)
    return smiles

def get_alphabet_from_tSMILES(smiles: list) -> list:
    tokens = []
    for smi in smiles:
        toks = seqs.get_tokens(smi)
        tokens.extend(toks)
    tokens = list(sorted(list(set(tokens))))
    return tokens
