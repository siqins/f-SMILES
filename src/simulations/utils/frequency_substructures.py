
from rdkit import Chem
from rdkit.Chem import AllChem

import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from enumerate_library import PATTERNS_SMA, PATTERNS_NAME

if __name__ == '__main__':

    path = r"C:\Users\12233\OneDrive\项目一\OPV\Smiles&HOMO_LUMO.xlsx"
    data = pd.read_excel(path, sheet_name='donor&acceptor')

    patterns = [Chem.MolFromSmarts(sma) for sma in PATTERNS_SMA]

    patterms_sma_dict = defaultdict(int)
    patterms_num_dict = defaultdict(int)

    for smi in tqdm(data['Smiles'].tolist()):
        try:
            mol = Chem.MolFromSmiles(smi)

            for name, sma in zip(PATTERNS_NAME, patterns):
                num = len(mol.GetSubstructMatches(sma))
                patterms_sma_dict[name] += num
                if num > 0:
                    patterms_num_dict[name] += 1
        except:
            pass

    patterms_sma = pd.DataFrame()
    for k, v in dict(patterms_sma_dict).items():
        patterms_sma[k] = [v]

    patterms_num = pd.DataFrame()
    for k, v in dict(patterms_sma_dict).items():
        patterms_num[k] = [v]

    patterms_sma.to_csv('sub_frequency.csv', index=False)
    patterms_num.to_csv('sub_numebrs.csv', index=False)