import os
import sys

current_file_path = os.path.realpath(__file__)
parent_path = os.path.dirname(current_file_path)
sys.path.append(parent_path)

import DataSet, MolUtils, Tools
from MolUtils import datamol
# import mol_utils.RDKUtils

from DataSet.STDTokens import CTokens, STDTokens_Frag_File

from MolUtils.RDKUtils.Frag.RDKFragUtil import Fragment_Alg
from DataSet.Graph.CNJMolAssembler import CNJMolAssembler
from DataSet.Graph.CNJMolUtil import CNJMolUtil
from DataSet.Graph.CNJTMol import CNJMolUtils