# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:30:02 2025

@author: dhuerta
In this file, I test the formulas for all partial derivatives of optical functions
with respect to quadrupole strength provided in the report, thus I can quantify
how bad with respect to each component the linear aproximation is.

In derivatives, the derivative with respect to que quadrupolar strength is the
first component.
Comprovem que el mètode, encara que comporta cert error, és correcte!
El problema abans sembla que estava en la manera de la qual calculava la 
"""

import at
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc #To set a cool font!
import useful_functions
import time
from joblib import Parallel, delayed
import copy



def calculate_numpy_ana_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, direction, divide):
    """
    
    Parameters
    ----------
    ring : at.lattice
        lattice for which the calculation is performed
    ind_bpm : tuple
        indices of the bpms
    ind_cor : tuple
        indices of the correctors
    ind_quad : tuple
        indices of the quadrupoles
    direction : char 
        'v' or 'h' the transeverse direction along which the ORM is calculated

    Returns
    -------
    The Jacobian of the Orbit response matrix object derivative with respect
    to the quadrupole strength calculated through Zeus formula.
    
    Part of the work is simply to merge the index lists to allow for a single
    call of the get_optics function providing better efficiency
    """
    
    dir_dict = {"h": 0, "v": 1}
    dir_ind = dir_dict[direction]
    sgn = -(-1)**dir_ind # 1 in vertical and -1 in horizontal
    
    ind_all  = np.array([[ind, 0] for ind in ind_bpm]
                       +[[ind, 1] for ind in ind_cor]
                       +[[ind, 2] for ind in ind_quad])
    
    ind_all  = ind_all[np.argsort(ind_all[:,0])]
    ind_dict = {
    "bpm":  [i for i, ind in enumerate(ind_all) if ind[1] == 0],
    "cor":  [i for i, ind in enumerate(ind_all) if ind[1] == 1],
    "quad": [i for i, ind in enumerate(ind_all) if ind[1] == 2]
    }
    
    ind_all=[ind[0] for ind in ind_all]
    start = time.perf_counter()
    
    allOptics = at.get_optics(ring, refpts=ind_all)
    
    print("Time to get all optics=", time.perf_counter()-start)
    
    start = time.perf_counter()
    tune = allOptics[1]["tune"][dir_ind]
    bpmBeta = np.array([allOptics[2]["beta"][ind][dir_ind] for ind in ind_dict["bpm"]])
    corBeta = np.array([allOptics[2]["beta"][ind][dir_ind] for ind in ind_dict["cor"]])
    #quadBeta= np.array([allOptics[2]["beta"][ind][dir_ind] for ind in ind_dict["quad"]])
    quadLen = np.array([ring[quad].Length for quad in ind_quad])#*(0.9988+dir_ind*0.0017)
    bpmTune = np.array([allOptics[2]["mu"][ind][dir_ind] for ind in ind_dict["bpm"]]) #Important, mu doesn't have the /2pi factor in atcollab!
    corTune = np.array([allOptics[2]["mu"][ind][dir_ind] for ind in ind_dict["cor"]]) 
    quadTune= np.array([allOptics[2]["mu"][ind][dir_ind] for ind in ind_dict["quad"]])
    
    #Now we use the lin_opt_6 method to compute the optic functions inside of quadrupoles
    
    #Objerve ho indices correspond like: k-> quadrupole, i->BPM, j-> corrector
    #These are the broadcasting variables used to write the tensor in numpy
    inQuadBeta0 = np.zeros([len(ind_quad), divide+1])
    inQuadTune0 = np.zeros([len(ind_quad), divide+1])
    
    for i, ind in enumerate(ind_quad):
        segment   = ring[0:0] + ring[ind].divide([1/divide]*(divide+1))
        #segment.enable_6d()
        segOptics = at.physics.linopt6(segment, refpts=range(divide+1), twiss_in=allOptics[2][ind_dict["quad"][i]])
        inQuadBeta0[i] = [j[dir_ind] for j in segOptics[2]["beta"]]
        inQuadTune0[i] = [j[dir_ind] + quadTune[i] for j in segOptics[2]["mu"]]
    
    inQuadBeta = np.zeros([len(ind_quad), divide])
    inQuadTune = np.zeros([len(ind_quad), divide])
    
    inQuadBeta = (inQuadBeta0[:,:-1] + inQuadBeta0[:,1:])/2
    inQuadTune = (inQuadTune0[:,:-1] + inQuadTune0[:,1:])/2
    
    
    
    bpmBetab = bpmBeta[None, :, None, None]
    corBetab = corBeta[None, None, :, None]
    quadBetab= inQuadBeta[:, None, None, :]
    quadLenb = quadLen[:, None, None, None]/(divide)
    bpmTuneb = bpmTune[None, :, None, None]
    corTuneb = corTune[None, None, :, None]
    quadTuneb= inQuadTune[:, None, None, :]
    
    print("Auxiliar Optics time = ", time.perf_counter()- start)
    start = time.perf_counter()
    Cij1 = np.cos(np.abs(bpmTuneb-corTuneb)-np.pi*tune)
    Cik2 = np.cos(2*np.abs(bpmTuneb-quadTuneb)-2*np.pi*tune)
    Cjk2 = np.cos(2*np.abs(corTuneb-quadTuneb)-2*np.pi*tune)
    Sij1 = np.sign(bpmTuneb-corTuneb)*np.sin(np.abs(bpmTuneb-corTuneb)-np.pi*tune)
    Sik2 = np.sign(bpmTuneb-quadTuneb)*np.sin(2*np.abs(bpmTuneb-quadTuneb)-2*np.pi*tune)
    Sjk2 = np.sign(corTuneb-quadTuneb)*np.sin(2*np.abs(corTuneb-quadTuneb)-2*np.pi*tune)
        
    cosTerm = Cij1 * ( Cik2 + Cjk2 + 2* np.cos(np.pi * tune)**2)
    sinTerm = Sij1 * ( Sik2 - Sjk2 + np.sin( 2*np.pi*tune)*(2*np.heaviside(bpmTuneb-quadTuneb, 0)
                -2*np.heaviside(corTuneb-quadTuneb, 0)-np.sign(bpmTuneb-corTuneb)))
    
    ana_dORM_dq = sgn * (
    np.sqrt(bpmBetab * corBetab) * quadBetab * quadLenb
    / (8 * np.sin(np.pi * tune)* np.sin(2 * np.pi * tune)) 
    * (cosTerm + sinTerm))
    print("Matrix time = ", time.perf_counter()- start)
    return np.sum(ana_dORM_dq, axis = 3)

#os.chdir('Z:\Projectes\ORM') #Set my working directory!
os.chdir('/Users/deumenec/Documents/Uni/9é semestre/ALBA/Teoria/ORM_compare') #Set my working directory!

###############################################################################
#
# Parameters to pass for the calculations
#
###############################################################################

lattice_file   = 'THERING.mat'
lattice_path   = 'lattices'
results        = 'results'
direction      = 'v'
step_exp       =  7
step           =  10**(-step_exp)
divide         =  10
#Steps for finite difference derivatives

compute_ana_ORM = False
#Compute the ORM analytically using the formula

compute_num_dORM = False
#To the numerical dORM_dq using tracking from ATcollab

#Plot parameters:

#Computes the dORM_dq by dividing quadrupoles into segments, calculates the ORM
#With respect to each of them and sums all the contributions into the actual ORM
#divide is the number of times each quadrupole is divided and just linopt is used
#to fet betas and phases in points other than the already used.
#Also finally, the average is used as we have seen it's better.

ring = at.load_mat(os.path.join(lattice_path, lattice_file))

ring.disable_6d()

        
ind_bpm  = np.array(at.get_refpts(ring, 'BPM'))  #Indices for the BPM
ind_cor  = np.array(at.get_refpts(ring, 'COR'))  #Indices for the correctors
ind_quad =  np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('QV') or el.FamName.startswith('QH')))

""" To select quadrupoles of each type
ind_quad = { 'v' : np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('QV'))),
             'h' : np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('QH')))}"""
ORM = analyticORM(ring, ind_bpm, ind_cor)

if (compute_num_dORM == True):
    dORM_num = calculate_num_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, step, direction)
    np.save(os.path.join(results,direction+ "_numdORM_dq"),dORM_num )
if (compute_num_dORM == False):
    dORM_num = np.load(os.path.join(results, direction+"_numdORM_dq.npy"))


dORM_numpy = calculate_numpy_ana_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, direction, divide)

#useful_functions.listPlot([dORM_num[16][40], dORM_numpy[16][40]], ["num", "numpy"], "titulinchi","proba")

vquadERROR = useful_functions.normalized_RMSE(dORM_num, dORM_numpy, (1,2))

vERROR = useful_functions.normalized_RMSE(dORM_num, dORM_numpy, (0,1,2))

length = 50
e_div_num = np.zeros(length-1)

for i in range(1, length):
    e_div_num[i-1] = useful_functions.normalized_RMSE(dORM_num, calculate_numpy_ana_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, direction, i), (0,1,2))
    
plt.plot(range(1, 50), e_div_num)
plt.show()




direction = "h"

if (compute_num_dORM == True):
    dORM_num = calculate_num_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, step, direction)
    np.save(os.path.join(results,direction+ "_numdORM_dq"),dORM_num )
if (compute_num_dORM == False):
    dORM_num = np.load(os.path.join(results, direction+"_numdORM_dq.npy"))

dORM_numpy = calculate_numpy_ana_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, direction, divide)

hquadERROR = useful_functions.normalized_RMSE(dORM_num, dORM_numpy, (1,2))

hERROR = useful_functions.normalized_RMSE(dORM_num, dORM_numpy, (0,1,2))
#Creating the plot Zeus asked me to:
   
quadBetas =np.array([ i[0] for i in (at.get_optics(ring, refpts=ind_quad))[2]["beta"] ])
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig, axis = plt.subplots(1,2,figsize=(10,5))
fig.suptitle("Errors along quadrupoles for the thin formula ", fontsize = 20)
fig.subplots_adjust(top=0.85)
plt.ylabel('dORM/dq normalized_RMSE \%')

quadBetash =np.array([ i[0] for i in (at.get_optics(ring, refpts=np.array(ind_quad)))[2]["beta"] ])
quadBetasv =np.array([ i[1] for i in (at.get_optics(ring, refpts=np.array(ind_quad)))[2]["beta"] ])

axis[0].set_xlabel('Quadrupole')
axis[1].set_xlabel('Quadrupole')
axis[0].title.set_text("Vertical direction, Total = "+f"{vERROR:.4f}\%")
axis[0].plot(vquadERROR)
#axis[0].plot((quadBetasv-3.85)/80+0.10)
axis[1].title.set_text("Horizontal direction, Total = "+f"{hERROR:.4f}\%" )
axis[1].plot(hquadERROR)
#axis[1].plot((quadBetash-3.85)/80+0.10)


plt.savefig("thin_dORM.pdf")
plt.show()
quad = 10
bpm = 10

