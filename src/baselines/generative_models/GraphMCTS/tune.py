import os
import time
import pickle
import pandas as pd
from rdkit import Chem
from src.GB_GM import scale_p_ring, generate_mol
from src.analyze_ZINC import read_file, get_mean_size, get_probs, operator
from src.analyze_ZINC import get_p, get_rxn_smarts_make_rings, get_rxn_smarts_rings
from src.analyze_ZINC import get_rxn_smarts, clean_probs, count_macro_cycles

path = os.path.dirname(__file__)

def analyze(smiles_list):

    elements = ['#5', '#6', '#7', '#8', '#9', '#14', '#15', '#16', '#17', '#35', '#53']
    bonds = ['-', '=', '#']


    mean_size, size_stdv = get_mean_size(smiles_list)
    print('mean number of non-H atoms', mean_size, '+/-', size_stdv)
    print('')

    smarts = ['[*]', '[R]', '[!R]', '[R2]']

    tot, probs = get_probs(smarts, smiles_list)
    print('Probability of ring atoms', float(probs['[R]']) / probs['[*]'])
    print('Probability of non-ring atoms', float(probs.get('[!R]', 0)) / probs['[*]'])
    print('Probability of fused-ring atoms', float(probs['[R2]']) / probs['[*]'])
    print('')

    smarts = ['[R]~[R]~[R]', '[R]-[R]-[R]', '[R]=[R]-[R]']
    tot, probs = get_probs(smarts, smiles_list, ring=True)

    print('Probability of [R]-[R]-[R]', float(probs['[R]-[R]-[R]']) / probs['[R]~[R]~[R]'])
    print('Probability of [R]=[R]-[R]', float(probs['[R]=[R]-[R]']) / probs['[R]~[R]~[R]'])
    print('')

    smarts = []
    for element in elements:
        smarts.append('[' + element + ']')

    smarts = []
    for element in elements:
        smarts.append('[' + element + 'R]')

    tot_Ratoms, probs_Ratoms = get_probs(smarts, smiles_list)

    R_elements = []
    for key in probs_Ratoms:
        R_elements.append(key)

    smarts = []

    for i, e1 in enumerate(R_elements):
        for e2 in R_elements:
            for j, e3 in enumerate(R_elements):
                if j >= i:
                    sm_s = e1 + '-' + e2 + '-' + e3
                    if sm_s not in smarts:
                        smarts.append(sm_s)
                sm_d = e1 + '=' + e2 + '-' + e3
                if sm_d not in smarts:
                    smarts.append(sm_d)

    tot, probs = get_probs(smarts, smiles_list, ring=True)

    sorted_x = sorted(probs.items(), key=operator.itemgetter(1), reverse=True)
    count = 0
    for i in range(len(sorted_x)):
        print(sorted_x[i][0], sorted_x[i][1] / tot)

    print('')

    rxn_smarts_rings = get_rxn_smarts_rings(probs)
    rxn_smarts_make_rings = get_rxn_smarts_make_rings(probs)
    p_rings = get_p(probs, tot)

    pickle.dump(p_rings, open('p_ring.p', 'wb'))
    pickle.dump(rxn_smarts_rings, open('rs_ring.p', 'wb'))
    pickle.dump(rxn_smarts_make_rings, open('rs_make_ring.p', 'wb'))

    smarts = []

    for bond in bonds:
        for element1 in elements:
            for element2 in elements:
                smarts.append('[' + element1 + ']' + bond + '[' + element2 + ';!R]')

    tot, probs = get_probs(smarts, smiles_list)
    tot, probs = clean_probs(probs)

    p = get_p(probs, tot)

    sorted_x = sorted(probs.items(), key=operator.itemgetter(1), reverse=True)
    count = 0
    for i in range(len(sorted_x)):
        print(sorted_x[i][0], sorted_x[i][1] / tot)

    rxn_smarts = get_rxn_smarts(probs)
    pickle.dump(p, open('p1.p', 'wb'))
    pickle.dump(rxn_smarts, open('r_s1.p', 'wb'))

    smarts_list = ['[*]1-[*]-[*]-1', '[*]1-[*]=[*]-1', '[*]1-[*]-[*]-[*]-1', '[*]1=[*]-[*]-[*]-1', '[*]1=[*]-[*]=[*]-1',
                   '[*]1-[*]-[*]-[*]-[*]-1', '[*]1=[*]-[*]-[*]-[*]-1', '[*]1=[*]-[*]=[*]-[*]-1',
                   '[*]1-[*]-[*]-[*]-[*]-[*]-1', '[*]1=[*]-[*]-[*]-[*]-[*]-1', '[*]1=[*]-[*]=[*]-[*]-[*]-1',
                   '[*]1=[*]-[*]-[*]=[*]-[*]-1', '[*]1=[*]-[*]=[*]-[*]=[*]-1']

    smarts_macro = ['[r;!r3;!r4;!r5;!r6;!r8;!r9;!r10;!r11;!r12]', '[r;!r3;!r4;!r5;!r6;!r7;!r9;!r10;!r11;!r12]',
                    '[r;!r3;!r4;!r5;!r6;!r7;!r8;!r10;!r11;!r12]', '[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r11;!r12]',
                    '[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r12]', '[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r11]']

    tot, probs = get_probs(smarts_list, smiles_list, ring=True)
    tot, probs = count_macro_cycles(smiles_list, smarts_macro, tot, probs)

    num_rings = 0
    print('')
    for key in probs:
        print(key, probs[key])
        num_rings += probs[key]

    print('')
    print('number of rings', num_rings)

def train(
        save_path,

        average_size = 26.37,
        size_stdev = 5.92,
        prob_double = 0.8,

        max_atoms = 50,

):

    p_ring = pickle.load(open(os.path.join(path, 'p_ring.p'), 'rb'))
    p_make_ring = p_ring
    rxn_smarts_make_ring = pickle.load(open(os.path.join(path, 'rs_make_ring.p'), 'rb'))
    rxn_smarts_ring_list = pickle.load(open(os.path.join(path, 'rs_ring.p'), 'rb'))

    rxn_smarts_list = pickle.load(open(os.path.join(path, 'r_s1.p'), 'rb'))
    p = pickle.load(open(os.path.join(path, 'p1.p'), 'rb'))

    p_ring = scale_p_ring(rxn_smarts_ring_list, p_ring, prob_double)
    p_make_ring = p_ring

    t0 = time.time()

    smiles = "CC"
    smiles = 'C1=CC=CS1'

    mol_list = []
    count = 1
    while count <= 10000:
        mol = generate_mol(smiles, max_atoms, average_size, size_stdev,
                           rxn_smarts_list, rxn_smarts_ring_list, rxn_smarts_make_ring, p, p_ring, p_make_ring)
        if mol is not None:
            new_smiles = Chem.MolToSmiles(mol)
            mol_list.append(new_smiles)
            print('id ', count, ' smi', new_smiles)

            count += 1

    t1 = time.time()

    print(t1 - t0)

    data = pd.DataFrame(mol_list, columns=['smiles'])
    data.to_csv(save_path)


if __name__ == '__main__':

    file_name = r'../../../../datasets/fused_units.csv'
    smis = pd.read_csv(file_name)['smiles'].tolist()
    analyze(smiles_list=smis)

    # train(save_path='results-2.csv')

    start_time = time.time()
    train(save_path='experiments/samples_2.csv')
    end_time = time.time()
    print(end_time - start_time)