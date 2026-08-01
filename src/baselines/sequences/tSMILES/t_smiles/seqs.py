from DataSet.STDTokens import CTokens, STDTokens_Frag_File
from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from DataSet.Graph.CNJMolAssembler import CNJMolAssembler
from DataSet.Graph.CNJMolUtil import CNJMolUtil
from DataSet.Graph.CNJTMol import CNJMolUtils

dec_algs = [Fragment_Alg.JTVAE]
ctoken = CTokens(STDTokens_Frag_File(None), max_length=1024, invalid=True, onehot=False)

def to_tsmiles(smi):
    dec_alg = dec_algs[0]
    combine_sml, combine_smt, amt_bfs_smarts = CNJMolUtils.encode_single(smi, ctoken, dec_alg)
    # print('[dec_alg is]:', dec_alg.name)
    # print('[TSSA/TSDY]:', combine_sml)
    # print('[TSID     ]:', combine_smt)
    # print('[TSIS     ]:', amt_bfs_smarts)
    return combine_sml


def from_tsmiles(combine_sml):
    bfs_ex = ''.join(combine_sml.strip().split(' '))
    n_samples = 1
    asm_alg = 'CALG_TSSA'
    # for i in range(n_samples):
    re_smils, bfs_ex_smiles_sub, new_vocs_sub, skt_wrong = CNJMolAssembler.decode_single(bfs_ex, ctoken, asm_alg, n_samples=1, p_mean=None)
    # print('dec_smile:=', re_smils)
    # print('bfs_ex_smiles_sub:=', bfs_ex_smiles_sub)
    # print('new_vocs_sub:=', new_vocs_sub)
    # print('skt_wrong:=', skt_wrong)
    return re_smils


def get_tokens(smiles):
    combine_sml = to_tsmiles(smiles)
    bfs_ex = ''.join(combine_sml.strip().split(' '))

    bfs_ex_smiles = CNJMolUtil.split_ex_smiles(bfs_ex, delimiter='^')

    n_samples = 1
    # asm_alg = 'CALG_TSSA'
    # for i in range(n_samples):
    # re_smils, bfs_ex_smiles_sub, new_vocs_sub, skt_wrong = CNJMolAssembler.decode_single(bfs_ex, ctoken, asm_alg, n_samples=1, p_mean=None)
    # print('dec_smile:=', re_smils)
    # print('bfs_ex_smiles_sub:=', bfs_ex_smiles_sub)
    # print('new_vocs_sub:=', new_vocs_sub)
    # print('skt_wrong:=', skt_wrong)
    # return new_vocs_sub[0]
    return bfs_ex_smiles


def get_tokens_from_tsmiles(tsmi):
    combine_sml = tsmi
    bfs_ex = ''.join(combine_sml.strip().split(' '))

    bfs_ex_smiles = CNJMolUtil.split_ex_smiles(bfs_ex, delimiter='^')

    n_samples = 1
    # asm_alg = 'CALG_TSSA'
    # for i in range(n_samples):
    # re_smils, bfs_ex_smiles_sub, new_vocs_sub, skt_wrong = CNJMolAssembler.decode_single(bfs_ex, ctoken, asm_alg, n_samples=1, p_mean=None)
    # print('dec_smile:=', re_smils)
    # print('bfs_ex_smiles_sub:=', bfs_ex_smiles_sub)
    # print('new_vocs_sub:=', new_vocs_sub)
    # print('skt_wrong:=', skt_wrong)
    # return new_vocs_sub[0]
    return bfs_ex_smiles


if __name__ == '__main__':
    smi = 'c1cc2sc3c([nH]c4c5[nH]c6c7sccc7sc6c5c5n[nH]nc5c43)c2s1'
    smi = 'C12=CC(CC3=C4SC5=C3SC=C5)=C4C=C1CC6=C2SC7=C6SC=C7'

    combine_sml = to_tsmiles(smi)
    from_tsmiles(combine_sml)
    print(get_tokens(smi))
