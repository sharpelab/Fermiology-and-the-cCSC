import os
import sys

# `sweep` is vendored under utils/sweep (a read-only subset of the lab
# `measureme` package); no external lab dependency is required.
from sweep import sweep_load as sl
from sweep import raster
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import colors
from cycler import cycler
from matplotlib.colors import Normalize
import matplotlib.image as mpimg

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, SymmetricalLogLocator, FormatStrFormatter
import matplotlib.patches as mpatches
from sklearn import linear_model


from functools import lru_cache
    
from utils import *
from Lf_analysis import Lf_analysis
# import cmcrameri.cm as cmc

# Physical constants
from scipy.constants import h, hbar, e, m_e
from scipy.constants import k as k_B
# h = 6.626E-34  # Planck constant (J·s)
# e = 1.6002E-19  # Elementary charge (C)
phi0 = h / e    # Flux quantum (Wb)
Rq = h/e**2
from matplotlib.gridspec import GridSpecFromSubplotSpec
file_path = '../data/raw_sweeps' # raw sweep data (download from SDR; see README)

norm_lf_delRxx = colors.SymLogNorm(linthresh=1, vmin=-1e3, vmax=1E3)
norm_psd = colors.LogNorm(vmin=5e-5, vmax=3e2) # In units of Ohms^2 technically


# update_rc_params / save_fig now live in utils.py (imported via the * above).


contact_colors = {
    '29-28': '#0173B2',
    '28-25': '#DE8F05',
    '25-24': '#CC78BC'
}
contact_colors_idx = {
    1: '#0173B2',
    2: '#DE8F05',
    3: '#CC78BC'
}
contact_names = {
    1: "B2-B3", 
    2: "B3-B4", 
    3: "B4-B5"
}
# contact_colors2 = {
#     '29-28': '#E69F00',
#     '28-25': '#56B4E9',
#     '25-24': '#CC79A7'
# }
# Densities for linecuts
density_targets = [0.45, 0.9] 
density_colors = ['#FF00FF', '#FF8000']