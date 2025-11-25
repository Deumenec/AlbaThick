# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:01:26 2025

@author: dhuerta

Functions written before for the ALBA lattice calculations
"""

import at
import numpy as np
import time
from joblib import Parallel, delayed
import copy

def get_beta_segment(ring, interval, num, direction, marks=[]):
    """
    This function returns the beta function along a series of num points in a segment and the used points
    It also plots the function showing elements underneath.
    
    Parameters
    ----------
    ring : at.lattice
        lattice under which the calculations are performed
    interval : [init, final]
        initial element and final element (both included)
    points : int
        number of points used
    direction : char
        "v" or "h" along which its beta is found and ploted
    marks : array
    Returns
    -------
    betas : np.array
    points: np.array
    """
    dir_dict = {"h": 0, "v": 1}
    dir_ind = dir_dict[direction]
    optics = at.get_optics(ring, refpts=range(interval[0], interval[1]+1))
    all_lengths = [element.Length for element in ring[0:(interval[1]+1)]]
    positions = [sum(all_lengths[0:i]) for i in range(interval[0], interval[1]+1)]
    lengths = all_lengths[interval[0]:interval[1]+1]
    points = np.linspace(positions[0], positions[-1],num)
    betas = np.zeros(num)
    #Amb un for i un while ho puc fer bastant òptimament!
    element = 0
    current_optics = optics[2][0]
    for i, point in enumerate(points):
        while(point>positions[element+1]):
            element +=1
            current_optics=optics[2][:][element]
        if (abs(point - positions[element])<10**-5): 
            betas[i] = current_optics["beta"][dir_ind]
        else:
            ring_segment = ring[element+interval[0]:element+interval[0]+1]
            new_elements = ring_segment[0].divide([(point-positions[element])/lengths[element], 1+(positions[element]-point)/lengths[element]])
            ring_segment[0] = new_elements[0]
            ring_segment.append(new_elements[1])
            newoptics = at.physics.linopt6(ring_segment, twiss_in=current_optics, refpts=1)
            betas[i] = newoptics[2]["beta"][0][dir_ind]
         #print(element)
    plt.plot(points,betas)
    for mark in marks:
        mark = mark-interval[0]
        xs = [positions[mark], positions[mark]]
        ys = [0,1]
        plt.plot(xs, ys, color = "red")
    plt.show()

def analyticORM(ring, ind_bpm, ind_cor):
    """Calculates analytically the ORM (this one has been checked to work)"""
    bpmOptics = at.get_optics(ring, refpts=ind_bpm)
    corOptics = at.get_optics(ring, refpts=ind_cor)
    tune = bpmOptics[1]["tune"][1]
    bpmBeta = [i[1] for i in bpmOptics[2]["beta"]]
    corBeta = [i[1] for i in corOptics[2]["beta"]]
    bpmTune = [i[1] for i in bpmOptics[2]["mu"]]
    corTune = [i[1] for i in corOptics[2]["mu"]]
    ORM = np.zeros([len(ind_bpm), len(ind_cor)])
    for i in range(len(ind_bpm)):
        for j in range(len(ind_cor)):
            ORM[i][j] = np.sqrt(bpmBeta[i]*corBeta[j])/(2*np.sin(np.pi*tune))*np.cos(abs(bpmTune[i]-corTune[j])-np.pi*tune)
    return np.array(ORM)

#Functions to calculate the dORMdq numerically.
def compute_single_quad(ring, quad, ORM, direction, step, ind_bpm, ind_cor):
    #Make a deep copy of the ring so threads don't interfere
    local_ring = copy.deepcopy(ring)

    #Change the quadrupole strength
    local_ring[quad].PolynomB[1] += step

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor
    )
    Resp_local.build_tracking()

    return (Resp_local.response - ORM) / step


def calculate_num_dORM_dq(ring, ind_bpm, ind_cor, ind_quad, step, direction):
    """
    
    Parameters
    ----------
    ring : at.lattice
        Ring for which the matrix is calculated
    ind_bpm : array
        indices of the BPMs for the ORM matrix
    ind_cor : array
        indices of the correctors for the ORM matrix
    ind_quad : array
        indices 
    dimension : char
        "v" for the vertical dimension and "h"
    Returns
    -------
    num_dORM_dq: np.array 
        The dORM_dq rank 3 tensor with indices dORM_dq[quadrupole][bpm][corrector]
    """
    num_dORM_dq = np.zeros([len(ind_quad), len(ind_bpm), len(ind_cor)])
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor) #class for computing the ORM
    Resp.build_tracking()
    ORM = Resp.response
    num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_single_quad) (ring, quad, ORM, direction, step, ind_bpm, ind_cor) for quad in ind_quad)
    num_dORM_dq = np.array(num_dORM_dq)
    
    return num_dORM_dq

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
    #Linopt 6 will be 6D (4D) if the ring is 6D (4D)
    
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






