import numpy as np
from rdkit import Chem
from typing import List
from itertools import chain


def get_products(products_list: List[List[Chem.Mol]]) -> List[Chem.Mol]:
    unique_smi = set(Chem.MolToSmiles(mol) for mol in chain.from_iterable(products_list))
    # we need to make a list from the set to be able to sort it to ensure reproducibility
    unique_smi = list(unique_smi)
    unique_smi = [smi.replace('~', '') for smi in unique_smi]
    unique_smi.sort()
    mols = [Chem.MolFromSmiles(smi) for smi in unique_smi if Chem.MolFromSmiles(smi)]

    if len(mols) != 0:
        mol = np.random.choice(mols)
    else:
        mol = None
    return mol


def are_tuples_connected(base_tuple, tuple1, tuple2):

    if not all(isinstance(t, tuple) for t in (base_tuple, tuple1, tuple2)):
        raise TypeError("The input must be a tuple")

    base_list = list(base_tuple)

    try:
        n = len(tuple1)
        idx1 = None
        for i in range(len(base_list) - n + 1):
            if tuple(base_list[i:i + n]) == tuple1:
                idx1 = i
                break

        m = len(tuple2)
        idx2 = None
        for j in range(len(base_list) - m + 1):
            if tuple(base_list[j:j + m]) == tuple2:
                idx2 = j
                break

        if idx1 is None or idx2 is None:
            return False

        if idx1 + n == idx2:
            return True
        elif idx2 + m == idx1:
            return True
        elif (idx1 == 0 and idx2 + m == len(base_list)) or (idx2 == 0 and idx1 + n == len(base_list)):
            return True
        else:
            return False

    except Exception as e:
        print(f"Exception: {e}")
        return False


def find_outermost_braces(s):
    stack = []
    result = []
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    result.append((start, i))
    outermost_braces = [s[start + 1:i] for start, i in result]
    parent_idx = [start - 1 for start, _ in result]
    return outermost_braces, parent_idx


def concat(dict_1, dict_2):
    all_keys = set(dict_1.keys()).union(set(dict_2.keys()))

    result = {}
    for key in all_keys:
        list_a = dict_1.get(key, [])
        list_b = dict_2.get(key, [])
        merged_list = list(set(list_a + list_b))
        result[key] = merged_list
    return result

