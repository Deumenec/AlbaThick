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
import numpy as np
import at
from scipy.io import loadmat
import at_utils

#os.chdir('Z:\Projectes\AlbaThick') #Set my working directory!
#os.chdir('/Users/deumenec/Documents/Uni/9é semestre/ALBA/Teoria/AlbaThick') #Set my working directory!

###############################################################################
#
# Parameters to pass for the calculations
#
###############################################################################

lattice_file   = 'ring_a2.mat'
lattice_folder = 'lattices'
results        = 'results'
direction      = 'v' #v: vertical h: horizontal b: both
step_exp       =  7
step           =  10**(-step_exp)
divide         =  10
read_numerical =  False


lat_path = os.path.join(lattice_folder, lattice_file)

ring = at.load_mat(lat_path, use = "ring")
mat         = loadmat(lat_path)

ind_bpm     = np.array([i[0]-1 for i in mat["bpmlist"]])
ind_cor     = { "v": np.array([i[0]-1 for i in mat["cmlist_v"]]), 
                "h": np.array([i[0]-1 for i in mat["cmlist_h"]])}
ind_quad    = np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('LIUQ') 
                                                   or el.FamName.startswith('LIDQ')
                                                   or  el.FamName.startswith('LQ') 
                                                   or el.FamName.startswith('MQ') 
                                                   or el.FamName.startswith('SQ')))



if read_numerical == False:
    #I add kick angle variable to perform the numerical ORM calculation
    sub_direction = "v"
    for ind in ind_cor[sub_direction]: ring[ind].KickAngle = np.array([0,0])
    numerical_ORM = at_utils.calc_numerical_dORM_dq(ring, ind_bpm, ind_cor[sub_direction], ind_quad, step, sub_direction)
    np.save(os.path.join(results,sub_direction+ "_numdORM_dq"),numerical_ORM)
    
    sub_direction = "h"
    for ind in ind_cor[sub_direction]: ring[ind].KickAngle = np.array([0,0])
    numerical_ORM = at_utils.calc_numerical_dORM_dq(ring, ind_bpm, ind_cor[sub_direction], ind_quad, step, sub_direction)
    np.save(os.path.join(results,sub_direction+ "_numdORM_dq"),numerical_ORM)
    






