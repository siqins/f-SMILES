
import shutil
import pandas as pd

from tqdm import tqdm
from rdkit import Chem

from pathlib import Path

from utils.enumerate_library import FRAGMENTS_SMI, REACTIONS_SMA, FRAGMENTS_CODE, FRAGMENTS_FREQUENCY
from utils.enumerate_library import PAHsBuilder

if __name__ == '__main__':

    mean_nRing = 11              # PAS环的平均个数
    max_nRing = 13              # PAS环的最大个数
    nmols = 100              # 生成PAS分子个数
    outname = 'gen'             # 结果文件名称

    DATA_FOLDER = Path('data')
    ENUM_FOLDER = DATA_FOLDER / 'enum'
    ETKDG_FOLDER = DATA_FOLDER / 'conf-gen'
    XTB_FOLDER = DATA_FOLDER / 'xtb-calc'

    ENUM_FOLDER.mkdir(parents=True, exist_ok=True)
    ETKDG_FOLDER.mkdir(parents=True, exist_ok=True)
    XTB_FOLDER.mkdir(parents=True, exist_ok=True)

    Builder = PAHsBuilder(mean_nRing=mean_nRing,
                          max_nRing=max_nRing,
                          fragments=FRAGMENTS_SMI,
                          reactions=REACTIONS_SMA,
                          fragments_code=FRAGMENTS_CODE,
                          fragments_frequency=FRAGMENTS_FREQUENCY)

    pbar = tqdm(total=nmols)
    mols = []
    for i in range(nmols):
        pbar.update(n=1)
        mol = Builder.generate_mol()
        if mol is not None:
            mols.append(mol)

    smis = [Chem.MolToSmiles(mol) for mol in mols]
    seqs = [mol.GetProp("seq") for mol in mols]
    names = [f"TM{i:06d}" for i in range(len(mols))]
    inchis = [Chem.MolToInchi(mol) for mol in mols]

    df = pd.DataFrame({"name": names, "smi": smis, "seq": seqs, "inchi": inchis})
    # filter by inchi
    df = df.drop_duplicates(subset="inchi", keep="first")
    # print the number of duplicates
    print(f"Number of duplicates: {nmols - len(df)}\n\n")

    # save the dataframe to test-dataset.csv ignoring the index
    enum_csv = ENUM_FOLDER / "dataset.csv"
    df.to_csv(enum_csv, index=False)

    # with open(f"{outname}.smi", "w+") as f:
    #     for smi in smis: f.write(f"{smi}\n")
    # with open(f"{outname}.seq", "w+") as f:
    #     for seq in seqs: f.write(f"{seq}\n")
    # with open(f"{outname}.csv", "w+") as f:
    #     for smi, seq in zip(smis, seqs):
    #         f.write(f"{smi},{seq}\n")