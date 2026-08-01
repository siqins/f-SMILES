"""general utils"""
import json


def read_file(path):
    """Return a file into list"""
    with open(path, 'r') as f:
        data = f.read().splitlines()
    return data


def load_json_config(path):
    """Reutrn a json file into dict"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def onek_encoding_unk(x, allowable_set):
    """One-hot embedding"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: int(x == s), allowable_set))
