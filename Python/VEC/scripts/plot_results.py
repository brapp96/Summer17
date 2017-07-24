"""

Plots results of VEC comparing the performance between different algorithms
using CCR and NMI metrics

7/24/2017, Anuththari Gamage

"""

import os
import sys

# add parent directory to path to access src folder
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir_name not in sys.path:
        sys.path.append(parent_dir_name)

from src import VEClib

# File path of the .pkl files containing results and parameters used for test
datafile = parent_dir_name + '/results/pkls/fulltest.pkl'
paramfile = parent_dir_name + '/results/pkls/fulltestparams.pkl'

VEClib.plot_res(datafile, paramfile)
raw_input()
