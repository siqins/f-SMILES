import time
from rdkit import Chem
import re
import numpy as np

PATTERN =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(PATTERN)


def find_biggest_digits(smiles=str()):
    """
    find the biggest digits by iterating throught possible max digits and test if there is a match in string
    input: smiles (string)
    output: max_digit (integer)
    """

    ### iterate through max digits possible solution
    for i in range(1, 10):
        digit = 10 - i
        if smiles.find(str(digit)) != -1:
            return digit

    return 0


def get_semantic_mem_map(smiles=str(), SMILES_REGEX=re.compile) -> np.array:
    """
    This function will generate a semantic memory map of a SMILES, i.e the number of semantic feature open for every token.
    Semantic feature include  branches and  rings
        smiles (string) : a valid SMILES
        SMILES_REGEX (compiled regular expression): compiled regular expression used to tokenize SMILES
        bonds_set (set of strings): a set containing all the bonds tokens
    output :
        mem_map  (numpy array):  the semantic memory map of the input SMILES
    """

    ### declare local variables
    tokens_list = SMILES_REGEX.findall(smiles)
    mem_map = np.zeros(len(tokens_list), dtype=int)
    digit_set = set()

    ### iterate throught tokens
    for i, token in enumerate(tokens_list):
        if token == "(":
            mem_map[i] += 1
        elif token == ")":
            mem_map[i] -= 1
        elif token.isdigit():
            if token in digit_set:
                digit_set.remove(token)
                mem_map[i] -= 1
            else:
                digit_set.add(token)
                mem_map[i] += 1

    return mem_map.cumsum()


def get_ClearSMILES(smiles: str, nb_random: int, SMILES_REGEX: re.compile("")) -> str:
    """
    This function will get a set of ClearSMILES, this is a stochastic process, thus the number of ClearSMILES may vary.
    A high number of random search  should yield a stable number of ClearSMILES
    input :
        smiles (string): a valid SMILES
        nb_random (integer): the number of randomized kekule SMILES to generate, i.e the number of random search
        SMILES_REGEX (compiled regex):  regex pattern used to tokenize SMILES
    output:
    results dict with string as keys and  various type of values
        smiles and nb_random  input are passed in the result
        max digit (integer), indicating the lowest maximum digit found in random SMILES
        lowest_mem_score (float), indicates the lowest score obtained by one or more ClearSMILES, currently the mean of semantic memory per token
        nb_unique_random_smiles (integer), number of unique randomized kekule SMILES found by random search
        nb_lowest_max_digit_smiles (integer), number of randomized kekule SMILES with the lowest max digit found
        nb_equivalent_solution  (integer) , which indicate the number of solution which achieved the lowest memory score
        ClearSMILES_set, value is (string): concatenation of the ClearSMILES_set as a string using "_" as a separator
        all keys including keyword time contains the duration of each stage of the pipeline, and value associated are float

    """

    ### declare local variable
    mol = Chem.MolFromSmiles(smiles)
    lowest_digit_smiles_set = set()
    lowest_mem_score_smiles_set = set()
    results_dict = {
        "smiles": smiles,
        "nb_random": nb_random,
        "max_digit": 9,
        "lowest_mem_score": np.inf,
        "nb_unique_random_smiles": int,
        "nb_lowest_max_digit_smiles": int,
        "nb_equivalent_solution": 0,
        "ClearSMILES_set": str,
        "random_gen_time": float,
        "min_max_digit_time": float,
        "mem_map_time": float,
        "total_time": time.perf_counter(),  # neglect the time to instantiate mol + empty sets
    }

    ### generate randomized kekule SMILES
    start_time = time.perf_counter()
    randomized_kekule_smiles_set = set(Chem.MolToRandomSmilesVect(mol, nb_random, kekuleSmiles=True))
    results_dict["nb_unique_random_smiles"] = len(randomized_kekule_smiles_set)
    results_dict["random_gen_time"] = time.perf_counter() - start_time

    ### find SMILES with lowest maximum digit
    start_time = time.perf_counter()
    for rd_smiles in randomized_kekule_smiles_set:

        ### declare loop variable
        temp_max_digit = find_biggest_digits(rd_smiles)

        ### if a new minimum is reach clear the set
        if temp_max_digit < results_dict["max_digit"]:
            lowest_digit_smiles_set.clear()
            results_dict["max_digit"] = temp_max_digit

        ### discard smiles if above current threshold
        elif temp_max_digit > results_dict["max_digit"]:
            continue

        lowest_digit_smiles_set.add(rd_smiles)
    results_dict["nb_lowest_max_digit_smiles"] = len(lowest_digit_smiles_set)
    results_dict["min_max_digit_time"] = time.perf_counter() - start_time

    ### Keep SMILES with lowest maximum digit for which the semantic memory score is minimal
    start_time = time.perf_counter()
    for ld_smiles in lowest_digit_smiles_set:

        ### declare loop variable
        mem_map = get_semantic_mem_map(ld_smiles, SMILES_REGEX)
        temp_mem_score = np.mean(mem_map)

        ### if a new minimum is reached, clear the set
        if temp_mem_score < results_dict["lowest_mem_score"]:
            lowest_mem_score_smiles_set.clear()
            results_dict["lowest_mem_score"] = temp_mem_score

        ### discard SMILES if mem score threshold is passed
        elif temp_mem_score > results_dict["lowest_mem_score"]:
            continue

        lowest_mem_score_smiles_set.add(ld_smiles)
    results_dict["mem_map_time"] = time.perf_counter() - start_time

    ### concatenate found ClearSMILES as string
    results_dict["ClearSMILES_set"] = "_".join(lowest_mem_score_smiles_set)
    results_dict["nb_equivalent_solution"] = len(lowest_mem_score_smiles_set)

    ### compute total time duration
    results_dict["total_time"] = time.perf_counter() - results_dict["total_time"]

    # return results_dict['']
    return results_dict["ClearSMILES_set"].split('_')[0]

if __name__ == "__main__":

    results_dict = get_ClearSMILES("CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", nb_random=1_00_000,
                                   SMILES_REGEX=SMILES_REGEX)
    print(results_dict)