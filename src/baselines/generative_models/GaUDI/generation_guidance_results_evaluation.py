
import os
from importlib import import_module

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import time

from typing import List


# disable rdkit warnings
rdkit.RDLogger.DisableLog('rdApp.*')

REACTIONS_EG_SMA = [
    "[#6:2]~1~[#6:5]~[#6;H2:4]~[#6;H1:3]~[#6:1]~1.[I]~[*:6]>>[#6:2]~1~[#6:5]~[#6;H2:4]~[#6:3](-[*:6])~[#6:1]~1",  # cyclopentadiene
    "[c:2]:1:[c:5]:[c;H:4]:[n;H:3]:[c:1]:1.[I]~[*:6]>>[c:2]:1:[c:5]:[c:4](-[*:6]):[n;H:3]:[c:1]:1",  # pyrrole
    "[c:2]:1:[c:5]:[c;H:4]:[o:3]:[c:1]:1.[I]~[*:6]>>[c:2]:1:[c:5]:[c:4](-[*:6]):[o:3]:[c:1]:1",  # furan
    "[#6:2]~1~[#6:5]~[#14;H2:4]~[#6;H1:3]~[#6:1]~1.[I]~[*:6]>>[#6:2]~1~[#6:5]~[#14;H2:4]~[#6:3](-[*:6])~[#6:1]~1",  # silole
    "[c:2]:1:[c:5]:[c;H:4]:[s:3]:[c:1]:1.[I]~[*:6]>>[c:2]:1:[c:5]:[c:4](-[*:6]):[s:3]:[c:1]:1",  # thiophene
    "[c:2]:1:[c:5]:[c;H:4]:[se:3]:[c:1]:1.[I]~[*:6]>>[c:2]:1:[c:5]:[c:4](-[*:6]):[se:3]:[c:1]:1",  # selenophene
]

REACTIONS_SC_SMA = [
    "[#6:2]~1~[#6:5]~[#6;H2:4]~[#6:3]~[#6:1]~1.[I]~[*:6]>>[#6:2]~1~[#6:5]~[#6:4](-[*:6])(-[*:6])~[#6:3]~[#6:1]~1",  # cyclopentadiene
    "[c:2]:1:[c:5]:[c:4]:[n;H:3]:[c:1]:1.[I]~[*:6]>>[c:2]:1:[c:5]:[c:4]:[n:3](-[*:6]):[c:1]:1",  # pyrrole
    "[C]>>[C]",  # furan
    "[#6:2]~1~[#6:5]~[#14;H2:4]~[#6:3]~[#6:1]~1.[I]~[*:6]>>[#6:2]~1~[#6:5]~[#14:4](-[*:6])(-[*:6])~[#6:3]~[#6:1]~1",  # silole
    "[C]>>[C]",  # thiophene
    "[C]>>[C]",  # selenophene
]

PATTERNS_EG_SMA = [
    "[#6:2]~1~[#6:5]~[#6;H2:4]~[#6;H1:3]~[#6:1]~1",  # cyclopentadiene
    "[c:2]:1:[c:5]:[c;H:4]:[n;H:3]:[c:1]:1",  # pyrrole
    "[c:2]:1:[c:5]:[c;H:4]:[o:3]:[c:1]:1",  # furan
    "[#6:2]~1~[#6:5]~[#14;H2:4]~[#6;H1:3]~[#6:1]~1",  # silole
    "[c:2]:1:[c:5]:[c;H:4]:[s:3]:[c:1]:1",  # thiophene
    "[c:2]:1:[c:5]:[c;H:4]:[se:3]:[c:1]:1",  # selenophene
]

PATTERNS_SC_SMA = [
    "[#6:2]~1~[#6:5]~[#6;H2:4]~[#6:3]~[#6:1]~1",  # cyclopentadiene
    "[c:2]:1:[c:5]:[c:4]:[n;H:3]:[c:1]:1",  # pyrrole
    "[c:2]:1:[c:5]:[c:4]:[o:3]:[c:1]:1",  # furan
    "[#6:2]~1~[#6:5]~[#14;H2:4]~[#6:3]~[#6:1]~1",  # silole
    "[c:2]:1:[c:5]:[c:4]:[s:3]:[c:1]:1",  # thiophene
    "[c:2]:1:[c:5]:[c:4]:[se:3]:[c:1]:1",  # selenophene
]

PATTERNS_NAME = [
    "cyclopentadiene",
    "pyrrole",
    "furan",
    "silole",
    "thiophene",
    "selenophene",
]

FRAGMENTS_SMI = [
    "C1=CCC=C1",  # cyclopentadiene
    "c1cc[nH]c1",  # pyrrole
    "c1ccoc1",  # furan
    "C1=C[SiH2]C=C1",  # silole
    "c1ccsc1",  # thiophene
    "c1cc[se]c1",  # selenophene
]

FRAGMENTS_CODE = [
    "cyclopentadiene",
    "pyrrole",
    "furan",
    "silole",
    "thiophene",
    "selenophene",
]


class AcceptorsBuilder(object):
    def __init__(self,
                 fragments: List[str] = None,
                 reactions_eg: List[str] = None,
                 reactions_sc: List[str] = None,
                 fragments_code: List[str] = None,
                 fragments_eg_sma: List[str] = None,
                 fragments_sc_sma: List[str] = None,
                 end_group=None,
                 side_chain=None,
                 random_seed=None):

        self.fragments = fragments
        self.fragment_code = fragments_code
        self.fragments_eg_sma = [Chem.MolFromSmarts(sma) for sma in fragments_eg_sma]
        self.fragments_sc_sma = [Chem.MolFromSmarts(sma) for sma in fragments_sc_sma]

        self.fragments_dict = {code: Chem.MolFromSmiles(smi) for smi, code in zip(fragments, fragments_code)}
        self.reactions_eg_dict = {code: AllChem.ReactionFromSmarts(sma) for sma, code in zip(reactions_eg, fragments_code)}
        self.reactions_sc_dict = {code: AllChem.ReactionFromSmarts(sma) for sma, code in zip(reactions_sc, fragments_code)}

        if random_seed is not None:
            self.seed = random_seed
            np.random.seed(random_seed)
        else:
            self.seed = None

        self.end_group = Chem.MolFromSmiles('O=C1C2=C(C=C(F)C(F)=C2)C(/C1=C/I)=C(C#N)\C#N') if end_group is None else end_group
        self.side_chain = Chem.MolFromSmiles('ICC(CCCC)CCCCCC') if side_chain is None else side_chain

    def generate_mol(self, mol) -> Chem.Mol:

        # adding end groups
        for code, fragment in zip(self.fragment_code, self.fragments_eg_sma):
            num_fragment = len(mol.GetSubstructMatches(fragment))
            # print(code, '  ', num_fragment)
            if num_fragment > 0:
                rxn = self.reactions_eg_dict[code]
                for _ in range(num_fragment):
                    mol = rxn.RunReactants([mol, self.end_group])[0][0]
                    mol.UpdatePropertyCache()

        # adding side chains
        for code, fragment in zip(self.fragment_code, self.fragments_sc_sma):
            if code in ["cyclopentadiene", "pyrrole", "silole"]:
                num_fragment = len(mol.GetSubstructMatches(fragment))
                # print(code, '  ', num_fragment)
                if num_fragment > 0:
                    rxn = self.reactions_sc_dict[code]
                    for _ in range(num_fragment):
                        mol = rxn.RunReactants([mol, self.side_chain])[0][0]
                        mol.UpdatePropertyCache()

        return mol


def main():
    Builder_IC_DH = AcceptorsBuilder(fragments=FRAGMENTS_SMI,
                                     reactions_eg=REACTIONS_EG_SMA,
                                     reactions_sc=REACTIONS_SC_SMA,
                                     fragments_code=FRAGMENTS_CODE,
                                     fragments_eg_sma=PATTERNS_EG_SMA,
                                     fragments_sc_sma=PATTERNS_SC_SMA,
                                     end_group=Chem.MolFromSmiles('O=C1C2=C(C(/C1=C/I)=C(C#N)/C#N)C=CC=C2'),
                                     side_chain=Chem.MolFromSmiles('ICC(CCCCCCCC)CCCCCC'), )

    Builder_IC2F_OB = AcceptorsBuilder(fragments=FRAGMENTS_SMI,
                                       reactions_eg=REACTIONS_EG_SMA,
                                       reactions_sc=REACTIONS_SC_SMA,
                                       fragments_code=FRAGMENTS_CODE,
                                       fragments_eg_sma=PATTERNS_EG_SMA,
                                       fragments_sc_sma=PATTERNS_SC_SMA)

    Builder_IC2Cl_HE = AcceptorsBuilder(fragments=FRAGMENTS_SMI,
                                        reactions_eg=REACTIONS_EG_SMA,
                                        reactions_sc=REACTIONS_SC_SMA,
                                        fragments_code=FRAGMENTS_CODE,
                                        fragments_eg_sma=PATTERNS_EG_SMA,
                                        fragments_sc_sma=PATTERNS_SC_SMA,
                                        end_group=Chem.MolFromSmiles('O=C1C2=C(C(/C1=C/I)=C(C#N)/C#N)C=C(Cl)C(Cl)=C2'),
                                        side_chain=Chem.MolFromSmiles('ICC(CCCC)CC'), )

    path = r"C:\Users\12233\Desktop\dataset.csv"
    save_path = ''
    data = pd.read_csv(path)['smi'].tolist()

    pbar = tqdm(total=len(data))
    smis_IC_DH, smis_IC2F_OB, smis_IC2Cl_HE = [], [], []
    for core in data:
        pbar.update(n=1)
        # print(core)
        core = Chem.MolFromSmiles(core)

        mol = Builder_IC_DH.generate_mol(core)
        try:
            sms = Chem.MolToSmiles(mol, kekuleSmiles=True)
        except:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            sms = Chem.MolToSmiles(mol, kekuleSmiles=True)
        smis_IC_DH.append(sms)

        mol = Builder_IC2F_OB.generate_mol(core)
        try:
            sms = Chem.MolToSmiles(mol, kekuleSmiles=True)
        except:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            sms = Chem.MolToSmiles(mol, kekuleSmiles=True)
        smis_IC2F_OB.append(sms)

        mol = Builder_IC2Cl_HE.generate_mol(core)
        try:
            sms = Chem.MolToSmiles(mol, kekuleSmiles=True)
        except:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            sms = Chem.MolToSmiles(mol, kekuleSmiles=True)
        smis_IC2Cl_HE.append(sms)

    data['acceptor_smi_IC_DH'] = smis_IC_DH
    data['acceptor_smi_IC2F_OB'] = smis_IC2F_OB
    data['acceptor_smi_IC2Cl_HE'] = smis_IC2Cl_HE
    data.to_csv(save_path, index=False)


    predict_model = import_module(r'D:\applications\les apps\OneDrive\项目一\OPV_performance\clac_opv\clac_opv.OPV_Calculator')

    path = 'best'
    files = os.listdir(path)

    pass

if __name__ == '__main__':
    main()