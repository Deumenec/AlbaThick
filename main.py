# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:14:40 2025

@author: dhuerta

In this file, I apply methods detailed by Zeus to calculate the Jacobian 
of the ORM better, After having calulated everything acording to the thin
dipole formula, I proceed to applying two new factors that must be acounted for:

Thick dipoles and dispersion, both of them essential as calculations for thin
dipoles have displayed. Numerical tests have also been performed to asses 
the effect over error 
"""

import os
import at
from scipy.io import loadmat

os.chdir('Z:\Projectes\AlbaThick') #Set my working directory!
#os.chdir('/Users/deumenec/Documents/Uni/9é semestre/ALBA/Teoria/AlbaThick') #Set my working directory!

###############################################################################
#
# Parameters to pass for the calculations
#
###############################################################################

lattice_file   = 'ring_a2.mat'
lattice_path   = 'lattices'
results        = 'results'
direction      = 'v'
step_exp       =  7
step           =  10**(-step_exp)
divide         =  10

file = os.path.join(lattice_path, lattice_file)

ring = at.load_mat(os.path.join(lattice_path, lattice_file), use = "ring")

mat = loadmat(file)
ind_bpm = mat["bpmlist"][0]
ind_corV= mat["cmlist_v"][0]
ind_corH= mat["cmlist_h"][0]