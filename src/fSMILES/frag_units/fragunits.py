import numpy as np
import pandas as pd
import csv, os
from rdkit import Chem
import molutils as F
from reactions import reactions_clear


class FragUnitCalc(object):

    def __init__(
            self,
            output: list=None,
            save_path: str=''
    ):
        if output is None:
            output = ['one_hot', 'one_hot_count', 'adjacent_matrix', 'node_matrix', 'index_data']

        self.output = output
        self.save_path = save_path
        if not os.path.exists(save_path) and save_path != '':
            os.makedirs(save_path)

    def compute(self, smis: list, names: list=None):

        smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]

        smi_list = []
        name_list = []
        for idx, smi in enumerate(smis):

            if '/' in smi:
                smi = smi.replace('/', '')
            if '\\' in smi:
                smi = smi.replace('\\', '')
            mol = Chem.MolFromSmiles(smi)
            if mol:
                smi = Chem.MolToSmiles(mol)
                smi_list.append(smi)
                if names is not None:
                    name_list.append(names[idx])
                else:
                    name_list.append(smi)

        ring_total_list = []
        error_find_independent_str1 = []
        total_neighbor_data = {}
        n = 0
        while n < len(smi_list):

            smiles = smi_list[n]
            name = name_list[n]

            from_get_bracket_index = F.get_bracket_index(smiles)
            left_index_list = from_get_bracket_index[0]
            right_index_list = from_get_bracket_index[1]
            index_list = from_get_bracket_index[2]

            cp_list = F.pairing(smiles, index_list, left_index_list, right_index_list)
            cp_arr = np.array(cp_list)
            index_arr = np.array(index_list)

            smallest_r = F.smallest(cp_list, index_arr)

            str_df = F.structure_DataFrame(cp_list, smallest_r, right_index_list, left_index_list)

            independent_cp_and_dependent_cp = F.rigin_type_classify(cp_list, smiles, smallest_r, str_df)
            independent_cp = independent_cp_and_dependent_cp[0]
            dependent_cp = independent_cp_and_dependent_cp[1]
            bratch = independent_cp_and_dependent_cp[3]
            bratch_cp = independent_cp_and_dependent_cp[2]

            cp_data = F.get_cp_data(cp_list, smallest_r, str_df, independent_cp, bratch_cp)

            find_str = F.find_independent_str(smiles, smallest_r, cp_data, independent_cp, dependent_cp, bratch_cp)
            string0 = find_str[0]
            index_data = find_str[1]
            index_cp = find_str[2]
            index_data0 = find_str[3]

            br = {}
            index_data2 = {}
            for k, v in index_data.items():
                j = v[1]
                j2 = F.add_bracket(j)
                j3 = F.make_smi(j2)
                j4 = F.link_c(j3)
                br_f = F.bratch_in_string(j4)
                j5 = br_f[0]
                bratch1 = br_f[1]
                mol = Chem.MolFromSmiles(j5)
                if mol:
                    j6 = Chem.MolToSmiles(mol)
                br_f2 = F.bratch_in_string(j6)
                j7 = br_f2[0]
                bratch2 = bratch1 + br_f2[1]
                br[k] = bratch2
                v2 = [v[0], j7]
                index_data2[k] = v2
            make_con_data = F.make_con(index_data2, index_cp, br)
            index_data3 = make_con_data[0]
            index_data4 = F.delete_free_radical_in_index_data(index_data3)
            index_cp2 = make_con_data[1]
            br2 = make_con_data[2]
            br3 = F.bratch_amend(br2)
            for k, v in index_data4.items():
                ring_total_list.append(v[1])
            for k, v in br3.items():
                v2 = []
                for j in v:
                    mol = Chem.MolFromSmiles(j)
                    if mol:
                        smi = Chem.MolToSmiles(mol)
                        v2.append(smi)
                        ring_total_list.append(smi)
                br3[k] = v2
            neighbor_data = F.found_neighbor(br3, str_df, index_data3, index_cp2)
            neighbor_data2 = F.found_end_point_neighbour(smiles, neighbor_data, index_data3)

            for k, v in neighbor_data2.items():
                for k2, v2 in v['right_neighbor'].items():
                    if '[C]' in v2:
                        v2 = v2.replace('[C]', 'C')
                for k2, v2 in v['left_neighbor'].items():
                    if '[C]' in v2:
                        v2 = v2.replace('[C]', 'C')
                if '[C]' in v['self']:
                    v['self'] = v['self'].replace('[C]', 'C')
            total_neighbor_data[name] = neighbor_data2
            n = n + 1

        ring_total_list2 = []
        for i in set(ring_total_list):
            ring_total_list2.append(i)

        ring_arr = np.array(ring_total_list2)
        ring_df = pd.DataFrame(ring_arr)

        if self.save_path != '':
            ring_df.to_csv(os.path.join(self.save_path, 'ring_total_list.csv'))

            if 'one_hot' in self.output:
                self.get_one_hot_code(ring_total_list2, total_neighbor_data, name_list)
            if 'one_hot_count' in self.output:
                self.get_one_hot_code_count(ring_total_list2, total_neighbor_data, name_list)
            if 'adjacent_matrix' in self.output:
                self.get_adjacent_matrix(ring_total_list2, total_neighbor_data, name_list)
            if 'node_matrix' in self.output:
                self.get_node_matrix(ring_total_list2, total_neighbor_data, name_list)
            if 'index_data' in self.output:
                self.get_index_data(ring_total_list2, total_neighbor_data, name_list)
        else:
            return ring_df

    def get_one_hot_code(self, ring_total_list2, total_neighbor_data, name_list):
        ring_series = pd.Series(ring_total_list2)

        m = 0
        for k, v in total_neighbor_data.items():
            if len(v) > m:
                m = len(v)
        long = len(ring_series)
        fp_list = []
        for i in name_list:
            fp1 = np.zeros((1, long))
            data = total_neighbor_data[i]
            for k, v in data.items():
                sel = v['self']
                column = ring_series[ring_series.values == sel].index[0]
                fp1[:, column] = int(1)
            fp1 = fp1.tolist()[0]
            fp_list.append(fp1)

        fp_arr = np.array(fp_list)
        fp_df = pd.DataFrame(fp_arr, index=name_list)

        fp_df.to_csv(os.path.join(self.save_path, "one_hot.csv"))

    def get_one_hot_code_count(self, ring_total_list2, total_neighbor_data, name_list):

        ring_series = pd.Series(ring_total_list2)

        m = 0
        for k, v in total_neighbor_data.items():
            if len(v) > m:
                m = len(v)
        long = len(ring_series)
        fp_list = []
        for i in name_list:

            fp1 = np.zeros((1, long))
            data = total_neighbor_data[i]
            for k, v in data.items():
                sel = v['self']
                column = ring_series[ring_series.values == sel].index[0]
                fp1[:, column] = fp1[:, column] + int(1)
            fp1 = fp1.tolist()[0]
            fp_list.append(fp1)

        fp_arr = np.array(fp_list)
        fp_df = pd.DataFrame(fp_arr, index=name_list)

        fp_df.to_csv(os.path.join(self.save_path, "one_hot_count.csv"))

    def get_adjacent_matrix(self, ring_total_list2, total_neighbor_data, name_list):
        ring_series = pd.Series(ring_total_list2)

        m = 0
        for k, v in total_neighbor_data.items():
            if len(v) > m:
                m = len(v)
        long = len(ring_series)
        matrix_list = []
        for i in name_list:
            data = total_neighbor_data[i]
            matrix2 = np.zeros((m, m))
            self_list = []
            for k, v in data.items():
                self_list.append(k)
            j = 0
            while j < len(self_list):
                k = self_list[j]
                data2 = data[k]
                for k, v in data2['right_neighbor'].items():
                    index = self_list.index(k)
                    matrix2[j, index] = 1
                for k, v in data2['left_neighbor'].items():
                    index = self_list.index(k)
                    matrix2[j, index] = 1
                j = j + 1
            f_matrix2 = matrix2.flatten()
            matrix_list.append(f_matrix2)

        adjacent_matrix = np.array(matrix_list)
        adjacent_matrix_df = pd.DataFrame(adjacent_matrix, index=name_list)
        adjacent_matrix_df.to_csv(os.path.join(self.save_path, "adjacent_matrix.csv"))

    def get_node_matrix(self, ring_total_list2, total_neighbor_data, name_list):
        ring_series = pd.Series(ring_total_list2)
        m = 0
        for k, v in total_neighbor_data.items():
            if len(v) > m:
                m = len(v)
        long = len(ring_series)
        node_list = []
        for i in name_list:

            matrix1 = np.zeros((long, m))
            data = total_neighbor_data[i]
            column_list = []
            for k, v in data.items():
                sel = v['self']
                column = ring_series[ring_series.values == sel].index[0]
                column_list.append(column)
            j = 0
            while j < len(column_list):
                k = column_list[j]
                matrix1[k, j] = 1
                j = j + 1
            matrix1 = matrix1.T
            f_matrix1 = matrix1.flatten()
            node_list.append(f_matrix1)

        node_matrix = np.array(node_list)
        node_matrix_df = pd.DataFrame(node_matrix, index=name_list)
        node_matrix_df.to_csv(os.path.join(self.save_path, "node_matrix.csv"))

    def get_index_data(self, ring_total_list2, total_neighbor_data, name_list):

        ring_series = pd.Series(ring_total_list2)

        m = 0
        for k, v in total_neighbor_data.items():
            if len(v) > m:
                m = len(v)
        long = len(ring_series)
        total_index_list = []
        for i in name_list:

            data = total_neighbor_data[i]
            fp = np.full((1, m), 'none')[0]
            index_list = []
            for k, v in data.items():
                sel = v['self']
                index = ring_series[ring_series.values == sel].index[0]
                index_list.append(index)
            n = len(index_list)
            j = 0
            while j < n:
                fp[j] = index_list[j]
                j = j + 1
            total_index_list.append(fp)
        index_frame = pd.DataFrame(total_index_list, index=name_list)
        index_frame.to_csv(os.path.join(self.save_path, 'index_data.csv'))


def unit_classify(ring_df: pd.DataFrame) ->pd.DataFrame:

    ring_df.columns = ['smiles']

    # 按单环，双环，支链，稠环进行划分
    num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '%']
    ring_type = []
    for smi in ring_df['smiles']:
        num = []
        for j in smi:
            if j in num_list:
                num.append(j)
        if len(num) == 0:
            ring_type.append('bratch')
        elif len(num) == 2:
            ring_type.append('single')
        elif len(num) == 4:
            ring_type.append('double')
        elif len(num) > 4:
            ring_type.append('fused')
        else:
            ring_type.append('unknow')

    ring_df['ring_type'] = ring_type
    return ring_df


def get_fused_rings(mol: Chem.Mol) -> Chem.Mol:
    while reactions_clear['clear'].RunReactants([mol]):
        mol = reactions_clear['clear'].RunReactants([mol])[0][0]
        Chem.SanitizeMol(mol)

    while reactions_clear['clear_5rings'].RunReactants([mol]):
        mol = reactions_clear['clear_5rings'].RunReactants([mol])[0][0]
        Chem.SanitizeMol(mol)

    while reactions_clear['clear_6rings'].RunReactants([mol]):
        mol = reactions_clear['clear_6rings'].RunReactants([mol])[0][0]
        Chem.SanitizeMol(mol)

    while reactions_clear['clear_n'].RunReactants([mol]):
        mol = reactions_clear['clear_n'].RunReactants([mol])[0][0]
        Chem.SanitizeMol(mol)

    while reactions_clear['clear_bi_bonds'].RunReactants([mol]):
        mol = reactions_clear['clear_bi_bonds'].RunReactants([mol])[0][0]
        Chem.SanitizeMol(mol)

    while reactions_clear['clear_bi_bonds_o'].RunReactants([mol]):
        mol = reactions_clear['clear_bi_bonds_o'].RunReactants([mol])[0][0]
        Chem.SanitizeMol(mol)
    return mol
