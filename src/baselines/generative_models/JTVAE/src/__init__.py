"""__init__ file"""

import os
import sys

current_file_path = os.path.realpath(__file__)
parant_path = os.path.dirname(current_file_path)
sys.path.append(parant_path)