#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Solve a given moment matrix using various ways.
"""

from cvxopt import matrix, sparse, spmatrix

import sympy as sp
import numpy as np

import util
import ipdb


_debug_mmsolvers = False

def monomial_filter(mono, filter='even'):
    if filter is 'even':
        if _debug_mmsolvers and not mono==1:
            print str(mono) + ':\t' + str(all([(i%2)==0 for i in mono.as_poly().degree_list()]))
        return 1 if mono==1 else int(all([i%2==0 for i in mono.as_poly().degree_list()]))

def get_cvxopt_Gh(LM, sparsemat = True):
    """
    get the G and h corresponding to this localizing matrix

    """

    if sparsemat:
        G = sparse(LM.get_LMI_coefficients(), tc='d').trans()
    else:
        G = matrix(LM.get_LMI_coefficients(), tc='d').trans()

    num_rms = len(LM.row_monos)
    h = matrix(np.zeros((num_rms, num_rms)))

    return {'G':G, 'h':h}
    
def get_cvxopt_inputs(MM, constraints = None, sparsemat = True, filter = 'even', slack = 0):
    """
    if provided, constraints should be a list of sympy polynomials that should be 0.
    @params - constraints: a list of sympy expressions representing the constraints in the same 

    """

    # Many optionals for what c might be, not yet determined really
    if filter is None:
        c = matrix(np.ones((MM.num_matrix_monos, 1)))
    else:
        c = matrix([monomial_filter(yi, filter='even') for yi in MM.matrix_monos], tc='d')

    Anp,bnp = MM.get_Ab(constraints)
    #_, residual, _, _ = scipy.linalg.lstsq(Anp, bnp)
    b = matrix(bnp)

    indicatorlist = MM.get_LMI_coefficients()

    if sparsemat:
        G = [sparse(indicatorlist).trans()]
        A = sparse(matrix(Anp))
    else:
        G = [matrix(indicatorlist).trans()]
        A = matrix(Anp)

    num_row_monos = len(MM.row_monos)
    h = [matrix(np.zeros((num_row_monos,num_row_monos)))]    

    return {'c':c, 'G':G, 'h':h, 'A':A, 'b':b}


def solve_moments_with_constraints(symbols, constraints, deg, slack = 1e-3):
    """
    Solve using the moment matrix.
    Use @symbols with basis bounded by degree @deg.
    Also use the constraints.
    """
    M = MomentMatrix(deg, symbols, morder='grevlex')

    cin = M.get_cvxopt_inputs(constraints, slack = slack)
    sol = solvers.sdp(cin['c'], Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b'])
    return M, sol

def test_mmsolvers():
    # simple test to make sure things run
    from cvxopt import solvers
    print 'testing simple unimixture with a skipped observation, just to test that things run'
    x = sp.symbols('x')
    M = mm.MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    #constrs = [x-1.5, x**2-2.5, x**3-4.5, x**4-8.5]
    cin = get_cvxopt_inputs(M, constrs, slack = 1e-5)

    #import MomentMatrixSolver
    #print 'joint_alternating_solver...'
    #y,L = MomentMatrixSolver.sgd_solver(M, constrs, 2, maxiter=101, eta = 0.001)
    #y,X = MomentMatrixSolver.convex_projection_solver(M, constrs, 2, maxiter=2000) 
    #print y
    #print X


    gs = [3-x, 3+x]
    locmatrices = [mm.LocalizingMatrix(M, g) for g in gs]
    Ghs = [get_cvxopt_Gh(lm) for lm in locmatrices]

    Gs=cin['G'] + [Gh['G'] for Gh in Ghs]
    hs=cin['h'] + [Gh['h'] for Gh in Ghs]
    
    sol = solvers.sdp(cin['c'], Gs=cin['G'], \
                  hs=cin['h'], A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-3)

    import mmextractors
    print mmextractors.extract_solutions_lasserre(M, sol['x'], Kmax = 2)
    print 'true values are 1 and 2'
    
if __name__=='__main__':
    test_mmsolvers()
    
