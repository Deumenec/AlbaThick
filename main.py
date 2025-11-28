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
import re #Per les regular expresions

import at_utils
import plot_utils

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
direction      = 'v' #v: vertical h: horizontal
step_exp       =  7
step           =  10**(-step_exp)
divide         =  10
read_numerical =  True
dispersion     =  False

if dispersion  == True:
    dsname = "d_"
if dispersion  == False:
    dsname = "nd_" 


lat_path = os.path.join(lattice_folder, lattice_file)

ring = at.load_mat(lat_path, use = "ring")

if dispersion == False:
    ring.disable_6d()



ordsV = re.compile('^COR$|^SH[1-7][1-4]?$|^SV[246]');
ordsH = re.compile('^COR$|^SV[1-7][1-4]?$');

ind_bpm = at.get_refpts(ring, lambda el: el.FamName.startswith('BPM') and not el.FamName.startswith('BPM_')) #BPMs bons sense l'element nou
#ind_bpm     = np.array([i[0]-1 for i in mat["bpmlist"]])
ind_cor     = { "v": at.get_refpts(ring, lambda el: ordsV.search(el.FamName)), 
                "h": at.get_refpts(ring, lambda el: ordsH.search(el.FamName))}
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
    np.save(os.path.join(results,dsname + sub_direction+ "_numdORM_dq"),numerical_ORM)
    
    sub_direction = "h"
    for ind in ind_cor[sub_direction]: ring[ind].KickAngle = np.array([0,0])
    numerical_ORM = at_utils.calc_numerical_dORM_dq(ring, ind_bpm, ind_cor[sub_direction], ind_quad, step, sub_direction)
    np.save(os.path.join(results,dsname + sub_direction+ "_numdORM_dq"),numerical_ORM)
    


dORMV = np.load(os.path.join(results,dsname + "v_numdORM_dq.npy"))
dORM_numV = at_utils.calc_thick_ana_dORM_dq(ring, ind_bpm, ind_cor["v"], ind_quad, "v", divide)

dORMH = np.load(os.path.join(results,dsname + "h_numdORM_dq.npy"))
dORM_numH = at_utils.calc_thick_ana_dORM_dq(ring, ind_bpm, ind_cor["h"], ind_quad, "h", divide)

plot_utils.plot_both(dORMV, dORMH, dORM_numV, dORM_numH)

at.plot_beta




