#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Given a moment matrix and its values, extract solutions
"""

import numpy as np
import scipy as sc
import scipy.linalg # for schur decomp, which np doesnt have
import numpy.linalg # for its norm, which suits us better than scipy
import util

def dict_mono_to_ind(monolist):
    dict = {}
    for i,mono in enumerate(monolist):
        dict[mono]=i
    return dict

def extract_solutions_lasserre(MM, ys, Kmax=10, tol=1e-5):
    """
    extract solutions via (unstable) row reduction described by Lassarre and used in gloptipoly
    MM is a moment matrix, and ys are its completed values
    """
    M = MM.numeric_instance(ys)
    Us,Sigma,Vs=np.linalg.svd(M)
    #
    #ipdb.set_trace()
    count = min(Kmax,sum(Sigma>tol))
    # now using Lassarre's notation in the extraction section of
    # "Moments, Positive Polynomials and their Applications"
    T,Ut = util.srref(Vs[0:count,:])
    print 'the biggest singular value we are losing is %f' % Sigma[count]
    # inplace!
    util.row_normalize_leadingone(Ut)

    couldbes = np.where(Ut>0.9)
    ind_leadones = np.zeros(Ut.shape[0], dtype=np.int)
    for j in reversed(range(len(couldbes[0]))):
        ind_leadones[couldbes[0][j]] = couldbes[1][j]

    basis = [MM.row_monos[i] for i in ind_leadones]
    dict_row_monos = dict_mono_to_ind(MM.row_monos)

    #ipdb.set_trace()
    Ns = {}
    bl = len(basis)
    # create multiplication matrix for each variable
    for var in MM.vars:
        Nvar = np.zeros((bl,bl))
        for i,b in enumerate(basis):
            Nvar[:,i] = Ut[ :,dict_row_monos[var*b] ]
        Ns[var] = Nvar

    N = np.zeros((bl,bl))
    for var in Ns:
        N+=Ns[var]*np.random.randn()
    T,Q=scipy.linalg.schur(N)

    sols = {}

    quadf = lambda A, x : np.dot(x, np.dot(A,x))
    for var in MM.vars:
        sols[var] = [quadf(Ns[var], Q[:,j]) for j in range(bl)]
    #ipdb.set_trace()
    return sols

def extract_solutions_dreesen_proto(MM, ys, Kmax=10, tol=1e-5):
    """
    extract solutions dreesen's nullspace method
    this is the prototype implementation that does not match solutions!
    """
    M = MM.numeric_instance(ys)
    Us,Sigma,Vs=sc.linalg.svd(M)
    
    count = min(Kmax,sum(Sigma>tol))
    Z = Us[:,0:count]
    print 'the next biggest eigenvalue we are losing is %f' % Sigma[count]

    dict_row_monos = dict_mono_to_ind(MM.row_monos)
    
    sols = {}
    for var in MM.vars:
        S1 = np.zeros( (len(MM.row_monos), len(MM.row_monos)) )
        Sg = np.zeros( (len(MM.row_monos), len(MM.row_monos)) )
        # a variable is in basis of the current var if var*basis in row_monos
        basis = []
        i = 0
        for mono in MM.row_monos:
            if mono*var in MM.row_monos:
                basis.append(mono)
                basisind = dict_row_monos[mono]
                gind = dict_row_monos[mono*var]
                
                S1[i, basisind] = 1
                Sg[i, gind] = 1
                i += 1
        S1 = S1[0:i,:]
        Sg = Sg[0:i,:]
        
        A = Sg.dot(Z)
        B = S1.dot(Z)
        
        # damn this, cant i just have a GE solver that works for non-square matrices?
        __,__,P = np.linalg.svd(np.random.randn(count,i), full_matrices = False)
        
        sols[var] = sc.real(sc.linalg.eigvals(P.dot(A),P.dot(B))).tolist()
        
    return sols


def test_solution_extractors():
    import sympy as sp
    x = sp.symbols('x')
    M = mm.MomentMatrix(2, [x], morder='grevlex')
    ys = [1, 1.5, 2.5, 4.5, 8.5]
    sols = extract_solutions_lasserre(M, ys)
    print sols
    print 'true values are 1 and 2'
    
if __name__ == "__main__":
    test_solution_extractors()
