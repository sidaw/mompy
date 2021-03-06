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
import scipy.sparse # 
import numpy.linalg # for its norm, which suits us better than scipy

from collections import defaultdict
import ipdb

EPS = 1e-5

def problem_to_str(obj, gs = None, hs = None, plain = True):
    strret = ''
    gpresent = gs is not None and len(gs)>0
    hpresent = hs is not None and len(hs)>0
    if not plain:
        strret += 'Minimizing '
        strret += '$\mathcal{L}(%s)$' % sp.latex(obj)
        if gpresent or hpresent:
            strret += ('\n\nsubject to \t')
            if gpresent:
                for g in gs:
                    strret += ' $%s$, \t' % sp.latex(g)
                strret += '\n'
            if hpresent:
                for h in hs:
                    strret += '$\mathcal{L}(%s) = 0$, \t' % sp.latex(h)
            strret = strret.strip(',')
        else:
            strret += '\t subject to no constraints'
    else:
        strret += 'Minimizing '
        strret += ' L(%s) ' % str(obj)
        if gpresent or hpresent:
            strret += ('\nsubject to \t')
            if gpresent:
                for g in gs:
                    strret += ' %s, \t' % str(g)
                strret += '\n'
            if hpresent:
                for h in hs:
                    strret += '$L(%s) = 0$, \t' % str(h)
            strret = strret.strip(',')
        else:
            strret += '\t subject to no constraints'
    return strret

class Measure(object):
    """
    Class representing a K atomic measure
    """
    
    def __init__(self, variables):
        if type(variables) is not list and type(variables) is not tuple:
            raise TypeError('variables need to be a list, even for single vars')
        # list of variables
        self.vars = variables
        # list of dictionaries keyed by variables
        self.atoms = []
        # corresponding pis for those variables
        self.weights = []
        
    def integrate(self, expr):
        integral = 0.0
        for i,w in enumerate(self.weights):
            integral += w* expr.subs(zip(self.vars, self.atoms[i]))
        return integral

    def normalize(self):
        Z = float(sum(self.weights))
        for i in xrange(len(self.weights)):
            self.weights[i] = self.weights[i] / Z
    
    def __add__(self, other):
        if type(other) == Measure:
            self.weights.append(other.weights)
            self.atoms.append(other.atoms)
        elif type(other) == tuple:
            self.weights.append(other[0])
            if type(other[1]) == np.ndarray:
                self.atoms.append(other[1].tolist())
            elif type(other[1]) == list:
                self.atoms.append(other[1])
            elif type(other[1]) == int or type(other[1]) == float:
                self.atoms.append([other[1]])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return self
    
    def merge(self):
        return NotImplemented
        
            
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

    def __len__(self):
        """returns m for this m by m matrix"""
        return len(self.row_monos)

    def __get_rowofA(self, constr):
        """
        @param - constr is a polynomial constraint expressed as a sympy
        polynomial. constr is h_j in Lasserre's notation,
        and represents contraints on entries of the moment matrix.
        """
        Ai = np.zeros(self.num_matrix_monos)
        constrpoly = constr.as_poly()
        #ipdb.set_trace()
        for i,yi in enumerate(self.matrix_monos):
            try:
                Ai[i] = constrpoly.coeff_monomial(yi)
            except ValueError:
                Ai[i] = 0
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
        Bf = scipy.sparse.lil_matrix((lenys, rowsM*rowsM))
        for i,yi in enumerate(self.matrix_monos):
            indices = self.term_to_indices_dict[yi]
            Bf[i, indices] = 1
        return scipy.sparse.csr_matrix(Bf)
        
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
    
    def get_Ab_slack(self, constraints=None, abs_slack=1e-2, rel_slack=1e-2, slackvector=0):
        print 'slacks'
        print abs_slack
        num_constrs = len(constraints) if constraints is not None else 0
        Anp = np.zeros((num_constrs, self.num_matrix_monos))
        bnp = np.zeros((num_constrs,1))
        if constraints is not None:
            for i,constr in enumerate(constraints):
                Anp[i,:] = self.__get_rowofA(constr)
                
        Aslack = np.zeros((2*num_constrs+2, self.num_matrix_monos))
        bslack = np.zeros((2*num_constrs+2,1))

        Aslack[0:num_constrs,:] = Anp
        Aslack[num_constrs:-2,:] = -Anp
        Aslack[-1,0] = 1
        bslack[-1] = 1
        Aslack[-2,0] = -1
        bslack[-2] = -1
        #ipdb.set_trace()
        #Aslack[0:num_constrs,0] += np.abs(Aslack[0:num_constrs,0])*1e-2 
        #Aslack[num_constrs:-2,0] += np.abs(Aslack[num_constrs:-2,0])*1e-2
        bslack[0:num_constrs] += abs_slack + slackvector
        bslack[num_constrs:-2] += abs_slack + slackvector

        return Aslack, bslack
        
    def numeric_instance(self, ys, maxdeg = None):
        """
        assign the matrix_monos ys and return an np matrix
        @params - ys: a list of numbers corresponding to self.matrix_monos
        @params - maxdeg: cutoff the matrix at this degree
        """
        assert len(ys)==len(self.matrix_monos), 'the lengths of the moment sequence is wrong'
        
        G = self.get_LMI_coefficients()
        num_inst = np.zeros(len(self.row_monos)**2)
        for i,val in enumerate(ys):
            num_inst += -val*np.array(matrix(G[i])).flatten()
        num_row_monos = len(self.row_monos)
        mat = num_inst.reshape(num_row_monos,num_row_monos)

        if maxdeg is not None:
            deglist = [sp.poly(rm, self.vars).total_degree() for rm in self.row_monos]
            cutoffind = sum([int(d<=maxdeg) for d in deglist])
            mat = mat[0:cutoffind, 0:cutoffind]

        return mat

    def pretty_print(self, sol):
        """
        print the moment matrix in a nice format?
        """
        for i,mono in enumerate(self.matrix_monos):
            print '%s:\t%f\t' % (str(mono), sol['x'][i])


class LocalizingMatrix(object):
    '''
    poly_g is a polynomial that multiplies termwise with a basic
    moment matrix of smaller size to give the localizing matrices This
    class depends on the moment matrix class and has exactly the same
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
