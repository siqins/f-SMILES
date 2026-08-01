from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

reactions_clear = {
    'clear': AllChem.ReactionFromSmarts('[#6;R:1]-[!R:2]>>[R:1]'),

    'clear_5rings': AllChem.ReactionFromSmarts('[#6;R:1]-[*:2]1~[*;H:3]~[!#6:4]~[*;H:5]~[*;H:6]~1>>[#6;R:1]'),

    'clear_6rings': AllChem.ReactionFromSmarts('[#6;R:1]-[#6:2]1~[#6;H:3]~[#6;H:4]~[#6;H:5]~[#6;H:6]~[#6;H:7]~1>>[#6;R:1]'),

    'clear_n': AllChem.ReactionFromSmarts('[#7;R:1]-[#6;!R:2]>>[#7;H:1]'),

    'clear_bi_bonds': AllChem.ReactionFromSmarts('[#6;R:1]=[#6;!R:2]>>[#6:1]'),

    'clear_bi_bonds_o': AllChem.ReactionFromSmarts('[#6;R:1]=[#8;!R:2]>>[#6:1]'),
}