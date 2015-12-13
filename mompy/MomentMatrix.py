#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
classes and helper moment matrices and localizing matrices,
which takes contraints as input produce the right outputs
for the cvxopt SDP solver

Sketchy: tested using sympy 0.7.6 (the default distribution did not work)
and cvxopt
"""

#from __future__ import division
import sympy as sp
import numpy as np
from cvxopt import matrix, sparse, spmatrix
import sympy.polys.monomials as mn
from sympy.polys.orderings import monomial_key

import scipy.linalg # for schur decomp, which np doesnt have
import numpy.linalg # for its norm, which suits us better than scipy

from collections import defaultdict
import util
import ipdb

EPS = 1e-7

class Measure(object):
    def __init__():
        print 'measure'
        
class MomentMatrix(object):
    """
    class to handle moment matrices and localizing matrices.
    Contains functions to process constraints
    
    degree: max degree of the basic monomials corresponding to each row, so the
    highest degree monomial in the entire moment matrix would be twice
    the provided degree.
    """
    def __init__(self, degree, variables, morder='grevlex', monos=None):
        """
        @param degree - highest degree of the first row/column of the
        moment matrix
        @param variables - list of sympy symbols
        @param morder the monomial order lex, grlex, grevlex, ilex, igrlex, igrevlex
        """
        self.degree = degree
        self.vars = variables
        self.num_vars = len(self.vars)

        # this object is a list of all monomials
        # in num_vars variables up to degree degree
        if monos is None:
            rawmonos = mn.itermonomials(self.vars, self.degree)
        else:
            rawmonos = monos

        # the reverse here is a bit random..., but has to be done.
        # also try grlex sometimes
        self.row_monos = sorted(rawmonos,\
                                 key=monomial_key(morder, self.vars[::-1]))

        # expanded monos is the flattened moment matrix itself
        self.expanded_monos = []
        for yi in self.row_monos:
            for yj in self.row_monos:
                self.expanded_monos.append(yi*yj)

        # This list correspond to the actual variables in the sdp solver
        self.matrix_monos = sorted(set(self.expanded_monos),\
                                   key=monomial_key(morder, self.vars[::-1]))
        self.num_matrix_monos = len(self.matrix_monos)

        # mapping from a monomial to a list of indices of
        # where the monomial appears in the moment matrix
        self.term_to_indices_dict = defaultdict(list)
        for i,yi in enumerate(self.expanded_monos):
            self.term_to_indices_dict[yi].append(i)

    def __str__(self):
        return 'moment matrix for %d variables: %s' % (self.num_vars, str(self.vars))

    def __get_rowofA(self, constr):
        """
        @param - constr is a polynomial constraint expressed as a sympy
        polynomial. constr is h_j in Lasserre's notation,
        and represents contraints on entries of the moment matrix.
        """
        Ai = np.zeros(self.num_matrix_monos)
        coefdict = constr.as_coefficients_dict();
        
        for i,yi in enumerate(self.matrix_monos):
            Ai[i] = coefdict.get(yi,0)
        return Ai

    def get_LMI_coefficients(self):
        """
        M = sum_\alpha y_alpha B_alpha, this function returns a list of B_alphas, as sparse matrices
        """
        Balpha = []
        constterm = True
        for yi in self.matrix_monos:
            indices = self.term_to_indices_dict[yi]
            term = spmatrix(-1,[0]*len(indices), indices, size=(1,len(self.expanded_monos)), tc='d')
            Balpha += [term]
        return Balpha

    def get_Bflat(self):
        """ M_flattened = sum_i y_i Bflat_i
        """
        rowsM = len(self.row_monos)
        lenys = len(self.matrix_monos)
        # consider using sparse Bf
        Bf = np.zeros((lenys, rowsM*rowsM))
        for i,yi in enumerate(self.matrix_monos):
            indices = self.term_to_indices_dict[yi]
            Bf[i, indices] = 1
        return Bf
        
    def get_Ab(self, constraints=None, cvxoptmode=True):
        num_constrs = len(constraints) if constraints is not None else 0
        Anp = np.zeros((num_constrs+1, self.num_matrix_monos))
        bnp = np.zeros((num_constrs+1,1))
        if constraints is not None:
            for i,constr in enumerate(constraints):
                Anp[i,:] = self.__get_rowofA(constr)

        Anp[-1,0] = 1
        bnp[-1] = 1
    
        # Remove redundant equations
        if cvxoptmode:
            Q, R = scipy.linalg.qr(Anp, mode='economic')
            Anp = R
            bnp = Q.T.dot(bnp)

            # Remove zero rows
            idx = np.sum(abs(Anp), 1) > EPS
            Anp = Anp[idx, :]
            bnp = bnp[idx, :]

        return Anp, bnp
        
    def numeric_instance(self, ys):
        """
        assign the matrix_monos ys and return an np matrix
        """
        assert(len(ys)==len(self.matrix_monos))
        
        G = self.get_LMI_coefficients()
        num_inst = np.zeros(len(self.row_monos)**2)
        for i,val in enumerate(ys):
            num_inst += -val*np.array(matrix(G[i])).flatten()
        num_row_monos = len(self.row_monos)
        return num_inst.reshape(num_row_monos,num_row_monos)

    def pretty_print(self, sol):
        """
        print the moment matrix in a nice format?
        """
        for i,mono in enumerate(self.matrix_monos):
            print '%s:\t%f\t' % (str(mono), sol['x'][i])


class LocalizingMatrix(object):
    '''
    poly_g is a polynomial that multiplies termwise with a basic
    moment matrix of smaller size to give the localizing matrices
    
    This class depends on the moment matrix class and has exactly the same
    monomials as the base moment matrix. So the SDP variables still
    corresponds to matrix_monos
    '''

    def __init__(self, mm, poly_g, morder='grevlex'):
        """
        @param - mm is a MomentMatrix object
        @param - poly_g the localizing polynomial
        """
        self.mm = mm
        self.poly_g = poly_g
        self.deg_g = poly_g.as_poly().total_degree()
        #there is no point to a constant localization matrix,
        #and it will cause crash because how sympy handles 1
        assert(self.deg_g>0)
        
        # change this to be everything still in mm.monos post multiplication 
        rawmonos = mn.itermonomials(self.mm.vars, self.mm.degree-self.deg_g);
        self.row_monos = sorted(rawmonos,\
                                 key=monomial_key(morder, mm.vars[::-1]))
        self.expanded_polys = [];
        for yi in self.row_monos:
            for yj in self.row_monos:
                self.expanded_polys.append(sp.expand(poly_g*yi*yj))
        # mapping from a monomial to a list of indices of
        # where the monomial appears in the moment matrix
        self.term_to_indices_dict = defaultdict(list)
        for i,pi in enumerate(self.expanded_polys):
            coeffdict = pi.as_coefficients_dict()
            for mono in coeffdict:
                coeff = coeffdict[mono]
                self.term_to_indices_dict[mono].append( (i,float(coeff)) )
        
    def get_LMI_coefficients(self):
        """
        polynomial here is called g in Lasserre's notation
        and defines the underlying set K some parallel with
        MomentMatrix.get_LMI_coefficients. Except now expanded_monos becomes
        expanded_polys
        """
        Balpha = []
        for yi in self.mm.matrix_monos:
            indices = [k for k,v in self.term_to_indices_dict[yi]]
            values = [-v for k,v in self.term_to_indices_dict[yi]]
            Balpha += [spmatrix(values, [0]*len(indices), \
                                        indices, size=(1,len(self.expanded_polys)), tc='d')]
        return Balpha

    

if __name__=='__main__':
    # simple test to make sure things run
    from cvxopt import solvers
    print 'testing simple unimixture with a skipped observation, just to test that things run'
    x = sp.symbols('x')
    M = MomentMatrix(3, [x], morder='grevlex')
    constrs = [x-1.5, x**2-2.5, x**4-8.5]
    #constrs = [x-1.5, x**2-2.5, x**3-4.5, x**4-8.5]
    Ab = M.get_Ab(constrs)

    gs = [3-x, 3+x]
    locmatrices = [LocalizingMatrix(M, g) for g in gs]
    Ghs = [lm.get_cvxopt_Gh() for lm in locmatrices]
