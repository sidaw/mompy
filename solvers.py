#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Solve a given moment matrix using various ways.
"""

from cvxopt import matrix, sparse, spmatrix, spdiag, solvers
solvers.options['maxiters'] = 150
solvers.options['feastol'] = 1e-6
solvers.options['abstol'] = 1e-7
solvers.options['reltol'] = 1e-6
solvers.options['show_progress'] = False
EPS = 1e-7

import sympy as sp
import numpy as np
import scipy as sc
import util
import ipdb
import itertools
import cvxopt.solvers

from core import MomentMatrix
from core import LocalizingMatrix

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
    
def get_cvxopt_inputs(MM, constraints = None, slack = 0, sparsemat = True, filter = 'even'):
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
    Glnp,hlnp = MM.get_Ab_slack(constraints, abs_slack = slack, rel_slack = slack)
    hl =  matrix(hlnp)
    
    if sparsemat:
        G = [sparse(indicatorlist).trans()]
        A = sparse(matrix(Anp))
        Gl = sparse(matrix(Glnp))
    else:
        G = [matrix(indicatorlist).trans()]
        A = matrix(Anp)
        Gl = matrix(Glnp)

    num_row_monos = len(MM.row_monos)
    h = [matrix(np.zeros((num_row_monos,num_row_monos)))]

    return {'c':c, 'G':G, 'h':h, 'A':A, 'b':b, 'Gl':Gl, 'hl':hl}
 
def get_constraint_row_monos(MM, constr):
    Ai = np.zeros(len(MM.row_monos))
    coefdict = constr.as_coefficients_dict()
    for i,yi in enumerate(len(MM.row_monos)):
        Ai[i] = coefdict.get(yi,0)
    return Ai

def solve_ith_GMP(MM, objective, gs, hs, slack = 1e-6):
    """
    Generalized moment problem solver
    @param - objective: a sympy expression that is the objective
    @param - gs: list of contraints defining the semialgebraic set K,
    each g corresponds to localizing matrices.
    @param - hs: constraints on the moments,
     not needed for polynomial optimization.
    """
    cin = get_cvxopt_inputs(MM, hs, slack=slack)
    Bf = MM.get_Bflat()
    objcoeff,__ = MM.get_Ab([objective], cvxoptmode = False)
    
    locmatrices = [LocalizingMatrix(MM, g) for g in gs]
    Ghs = [get_cvxopt_Gh(lm) for lm in locmatrices]

    # list addition
    Gs=cin['G'] + [Gh['G'] for Gh in Ghs]
    hs=cin['h'] + [Gh['h'] for Gh in Ghs]
    # print Ghs
    solsdp = cvxopt.solvers.sdp(matrix(objcoeff[0,:]), Gs=Gs, hs=hs, A=cin['A'], b=cin['b'])
    
    #ipdb.set_trace()
    return solsdp



def solve_GMP(objective, gs, hs, rounds = None):
    """
    Outer loop of the generalized moment problem solver
    @param - objective: a sympy expression that is the objective
    @param - gs: list of contraints defining the semialgebraic set K,
    each g corresponds to localizing matrices.
    @param - hs: constraints on the moments,
     not needed for polynomial optimization.
    """

    # problem setup here
    constrs = []
    syms = objective.free_symbols
    mindeg = objective.as_poly().total_degree()
    for g in gs:
        if type(g) == sp.relational.GreaterThan:
            nf = g.lhs - g.rhs
        elif type(g) ==  sp.relational.LessThan:
            nf = g.rhs - g.lhs
        constrs += [nf]

        degnf = nf.as_poly().total_degree()
        if degnf > mindeg: mindeg = degnf

        syms.update(nf.free_symbols)
        
    print 'the maximum degree appearing in the problem is %d' % mindeg
    #for g in constrs:
    #    print g
    
    mdeg = int( (mindeg+1)/2 )
    if rounds is None:
        rounds = 2

    objvals = {}
    for i in range(mdeg,mdeg + rounds):
        MM = MomentMatrix(i, list(syms), morder='grevlex')
        soldict = solve_ith_GMP(MM, objective, constrs, hs)
        soldict['MM'] = MM
        r =  np.linalg.matrix_rank(MM.numeric_instance(soldict['x']), 1e-4)
        print 'round=%d out of %d,\t rank=%d,\t obj=%.3f' % (i-mdeg+1, rounds, r, soldict['primal objective'])
        objvals[i] = soldict['primal objective']
    return soldict

def solve_polyopt(objective, gs):
    pass

def solve_basic_constraints(MM, constraints, slack = 1e-2):
    """
    Solve using the moment matrix.
    Use @symbols with basis bounded by degree @deg.
    Also use the constraints.
    """
    cin = get_cvxopt_inputs(MM, constraints, slack=slack)
    Bf = MM.get_Bflat()
    R = np.random.rand(len(MM), len(MM))
    W = R.dot(R.T)
    #W = np.eye(len(MM))
    w = Bf.dot(W.flatten())
    solsdp = cvxopt.solvers.sdp(c=cin['c'], Gs=cin['G'], hs=cin['h'], Gl=cin['Gl'], hl=cin['hl'])
    #ipdb.set_trace()
    return solsdp

def solve_generalized_mom_coneqp(MM, constraints, pconstraints=None, maxiter = 1):
    """
    solve using iterative GMM using the quadratic cone program
    func_W takes a solved instance and returns the weighting matrix,
    this function has access to individual data points
    @params
    constraints - E[g(x,X)] = f(x) - h(X) that are supposed to be 0
    Eggt - the function handle takes current f(x) and estimates
    E[g(x,X)g(x,X)'] \in \Re^{n \times n}, the information matrix
    maxiter - times to run the iterative GMM
    """
    N = len(constraints)
    D = len(MM.matrix_monos)
    sr = len(MM.row_monos)
    
    A,b = MM.get_Ab(constraints, cvxoptmode = False)
    #ipdb.set_trace()
    # augumented constraint matrix introduces slack variables g
    A_aug = sparse(matrix(sc.hstack((A, 1*sc.eye(N+1)[:,:-1]))))
    P = spdiag([matrix(0*np.eye(D)), matrix(np.eye(N))])
    b = matrix(b)
       
    indicatorlist = MM.get_LMI_coefficients()
    G = sparse(indicatorlist).trans()
    V,I,J = G.V, G.I, G.J, 
    Gaug = sparse(spmatrix(V,I,J,size=(sr*sr, N + D)))
    h = matrix(np.zeros((sr*sr,1)))

    dims = {}
    dims['l'] = 0
    dims['q'] = []
    dims['s'] = [sr]

    Bf = MM.get_Bflat()
    R = np.random.rand(len(MM), len(MM))
    W = R.dot(R.T)
    W = np.eye(len(MM))
    w = Bf.dot(W.flatten())[:,np.newaxis]
    q = 1e-5*matrix(np.vstack( (w,np.zeros((N,1))) ))
    
    #ipdb.set_trace()
    for i in xrange(maxiter):
        w = Bf.dot(W.flatten())[:,np.newaxis]
        sol = solvers.coneqp(P, q, G=Gaug, h=h, dims=dims, A=A_aug, b=b)
    sol['x'] = sol['x'][0:D]
    return sol

def solve_generalized_mom_conelp(MM, constraints, W=None, absslack=1e-4, totalslack=1e-2, maxiter = 1):
    """
    solve using iterative GMM using the cone linear program
    W is a specific weight matrix

    we give generous bound for each constraint, and then harsh bound for
    g'Wg
    @params
    constraints - E[g(x,X)] = f(x) - phi(X) that are supposed to be 0
    Eggt - the function handle takes current f(x) and estimates
    E[g(x,X)g(x,X)'] \in \Re^{n \times n}, the information matrix
    maxiter - times to run the iterative GMM
    """
    
    N = len(constraints)
    D = len(MM.matrix_monos)
    sr = len(MM.row_monos)
    
    A,b = MM.get_Ab(constraints, cvxoptmode = False)
    # augumented constraint matrix introduces slack variables g
    A_aug = sparse(matrix(sc.hstack((A, 1*sc.eye(N+1)[:,:-1]))))
    P = spdiag([matrix(0*np.eye(D)), matrix(np.eye(N))])
    b = matrix(b)
       
    indicatorlist = MM.get_LMI_coefficients()
    G = sparse(indicatorlist).trans()
    V,I,J = G.V, G.I, G.J, 
    Gaug = sparse(spmatrix(V,I,J,size=(sr*sr, N + D)))
    h = matrix(np.zeros((sr*sr,1)))

    dims = {}
    dims['l'] = 0
    dims['q'] = []
    dims['s'] = [sr]

    Bf = MM.get_Bflat()
    R = np.random.rand(len(MM), len(MM))
    #W = R.dot(R.T)
    W = np.eye(len(MM))
    w = Bf.dot(W.flatten())[:,np.newaxis]
    q = matrix(np.vstack( (w,np.zeros((N,1))) ))
    
    #ipdb.set_trace()
    for i in xrange(maxiter):
        w = Bf.dot(W.flatten())[:,np.newaxis]
        sol = solvers.coneqp(P, q, G=Gaug, h=h, dims=dims, A=A_aug, b=b)
    sol['x'] = sol['x'][0:D]
    return sol


# solve for the weight matrix in convex iteration
def solve_W(Xstar, rank):
    """
    minimize trace(Xstar*W)
    s.t.  0 \preceq W \preceq I
    trace(W) = n - rank     
    """
    Balpha = []
    numrow = Xstar.shape[0]
    lowerdiaginds = [(i,j) for (i,j) in \
                     itertools.product(xrange(numrow), xrange(numrow)) if i>j]
    diaginds = [i+i*numrow for i in xrange(numrow)]
    for i in diaginds:
        indices = [i]
        values = [-1]
        Balpha += [spmatrix(values, [0]*len(indices), \
        indices, size=(1,numrow*numrow), tc='d')]
        
    for i,j in lowerdiaginds:
        indices = [i+j*numrow, j+i*numrow]
        values = [-1, -1]
        Balpha += [spmatrix(values, [0]*len(indices), \
        indices, size=(1,numrow*numrow), tc='d')]
        
    Gs = [sparse(Balpha, tc='d').trans()] + [-sparse(Balpha, tc='d').trans()]
    hs = [matrix(np.zeros((numrow, numrow)))] + [matrix(np.eye(numrow))]
    A = sparse(spmatrix([1]*numrow, [0]*numrow, \
        range(numrow), size=(1,numrow*(numrow+1)/2), tc='d'))
    b = matrix([numrow - rank], size=(1,1), tc='d')
    
    x = [np.sum(Xstar.flatten()*matrix(Balphai)) for Balphai in Balpha]
    sol = cvxopt.solvers.sdp(-matrix(x), Gs=Gs, hs=hs, A=A, b=b)
    w = sol['x']
    Wstar = 0
    for i,val in enumerate(w):
        Wstar += -val*np.array(Balpha[i])
    #ipdb.set_trace()
    return Wstar,sol
    
    
def solve_moments_with_convexiterations(MM, constraints, maxrank = 3, slack = 1e-3, maxiter = 200):
    """
    Solve using the moment matrix iteratively using the rank constrained convex iterations
    Use @symbols with basis bounded by degree @deg.
    Also use the constraints.
    """
    cin = get_cvxopt_inputs(MM, constraints)
    Bf = MM.get_Bflat()
    R = np.random.rand(len(MM), len(MM))
    #W = R.dot(R.T)
    W = np.eye(len(MM))
    tau = []
    for i in xrange(maxiter):
        w = Bf.dot(W.flatten())
        #solsdp = cvxopt.solvers.sdp(cvxopt.matrix(w), Gs=cin['G'], hs=cin['h'], A=cin['A'], b=cin['b'])
        solsdp = cvxopt.solvers.sdp(cvxopt.matrix(w), Gs=cin['G'], hs=cin['h'], Gl=cin['Gl'], hl=cin['hl'])
        Xstar = MM.numeric_instance(solsdp['x'])
        W,solW = solve_W(Xstar, maxrank)
        W = np.array(W)
        ctau =  np.sum(W * Xstar.flatten())
        if ctau < 1e-3:
            break
        tau.append(ctau)
        
    #ipdb.set_trace()
    print tau
    return solsdp

def test_mmsolvers():
    # simple test to make sure things run
    
    print 'testing simple unimixture with a skipped observation, just to test that things run'
    x = sp.symbols('x')
    M = MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    #constrs = [x-1.5, x**2-2.5, x**3-4.5, x**4-8.5]
    cin = get_cvxopt_inputs(M, constrs, slack = 1e-5)

    #import MomentMatrixSolver
    #print 'joint_alternating_solver...'
    #y,L = MomentMatrixSolver.sgd_solver(M, constrs, 2, maxiter=101, eta = 0.001)
    #y,X = MomentMatrixSolver.convex_projection_solver(M, constrs, 2, maxiter=2000) 
    #print y
    #print X

    gs = [2-x, 2+x]
    locmatrices = [LocalizingMatrix(M, g) for g in gs]
    Ghs = [get_cvxopt_Gh(lm) for lm in locmatrices]

    Gs=cin['G'] + [Gh['G'] for Gh in Ghs]
    hs=cin['h'] + [Gh['h'] for Gh in Ghs]
    
    sol = solvers.sdp(cin['c'], Gs=Gs, \
                  hs=hs, A=cin['A'], b=cin['b'])

    print sol['x']
    print abs(sol['x'][3]-4.5)
    assert(abs(sol['x'][3]-4.5) <= 1e-3)

    import extractors
    print extractors.extract_solutions_lasserre(M, sol['x'], Kmax = 2)
    print 'true values are 1 and 2'
    
if __name__=='__main__':
    test_mmsolvers()
    
