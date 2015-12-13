import sympy as sp
import numpy as np

import sympy.polys.monomials as mn
from sympy.polys.orderings import monomial_key

import scipy.linalg # for schur decomp, which np doesnt have
# import numpy.linalg # for its norm, which suits us better than scipy

from collections import defaultdict
import util
import ipdb
from cvxopt import matrix, sparse, spmatrix, solvers

EPS = 1e-7

def convex_projection_solver(M, constraints, rank=2, tau=1, delta = 0.1, maxiter=100, tol=1e-6):
    """
    does projection, so that L.T L is closest to M(y)
    and y is the projection of cloest to L.T L what satisfies the contraints
    any fixed point is a valid solution
    """
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    Bf = M.get_Bflat()
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)

    weightone = 1
    A = A*weightone; b = b*weightone
    X = np.random.randn(rowsM, rowsM)
    for i in xrange(maxiter):
        yX = Bf.dot(X.flatten()[:, np.newaxis])/np.sum(Bf,1)[:,np.newaxis]

        yproj = util.project_nullspace(A,b,yX, randomize =0)

        y = yproj #yX + delta*(yproj - yX)
        My = M.numeric_instance(y)
        objyconstrs = scipy.linalg.norm(X- M.numeric_instance(y)) + scipy.linalg.norm(A.dot(y)-b)
        
        U,D,V=scipy.linalg.svd(My)
        D = np.fmax(D - tau, 0)
        X = U.dot(np.diag(D)).dot(V)
        objprojL = scipy.linalg.norm(X - M.numeric_instance(y)) + scipy.linalg.norm(A.dot(y)-b)
        print '%d:\t%f\t%f' % (i, objyconstrs, objprojL)
        if objprojL < tol:
            break
    return y,X


def projection_solver(M, constraints, rank=2, maxiter=100, tol=1e-6):
    """
    does projection, so that L.T L is closest to M(y)
    and y is the projection of cloest to L.T L what satisfies the contraints
    any fixed point is a valid solution
    """
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    L = np.random.randn(rank, len(M.row_monos))
    L = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2) + np.random.randn(2,4)*0.3
    Bf = M.get_Bflat()
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)

    weightone = 1
    A = A*weightone; b = b*weightone
    
    for i in xrange(maxiter):
        yL = Bf.dot(L.T.dot(L).flatten()[:, np.newaxis])/np.sum(Bf,1)[:,np.newaxis]
        y = util.project_nullspace(A,b,yL, randomize =0/np.sqrt(i+100))
        My = M.numeric_instance(y)

        objyconstrs = scipy.linalg.norm(L.T.dot(L) - M.numeric_instance(y)) + scipy.linalg.norm(A.dot(y)-b)
        
        U,D,V=scipy.linalg.svd(My)
        L = V[0:rank,:]*np.sqrt(D[0:rank, np.newaxis])
        
        objprojL = scipy.linalg.norm(L.T.dot(L) - M.numeric_instance(y)) + scipy.linalg.norm(A.dot(y)-b)
        print '%d:\t%f\t%f' % (i, objyconstrs, objprojL)
        if objprojL < tol:
            break
    return y,L
    
    #ipdb.set_trace()
    # need to know which matrix_mono is in each location
    
def joint_alternating_solver(M, constraints, rank=2, maxiter=100, tol=1e-3):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    L = np.random.randn(rank, len(M.row_monos))
    L = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    Bf = M.get_Bflat()
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)

    weightone = 1
    A = A*weightone; b = b*weightone
    
    consts = np.zeros((lenLs + lenys, 1))
    consts[-lenys:] = A.T.dot(b)
    coeffs = np.zeros((lenLs+lenys,lenLs+lenys))

    for i in xrange(maxiter):
        smallblock = L.dot(L.T)
        dy_L = -Bf.dot(np.kron(np.eye(rowsM), L.T))
        dy_y = A.T.dot(A) + Bf.dot(Bf.T)

        dL_L = np.kron(np.eye(rowsM), smallblock)
        dL_y = -dy_L.T

        coeffs[0:lenLs, 0:lenLs] = dL_L
        coeffs[0:lenLs, lenLs:] = dL_y

        coeffs[lenLs:, 0:lenLs] = dy_L
        coeffs[lenLs:, lenLs:] = dy_y
        
        sol = scipy.linalg.solve(coeffs, consts)
        
        L = sol[0:lenLs].reshape((rank, rowsM))

        # hack to normalize
        # L = L / scipy.linalg.norm(L[:,0])
        y = sol[-lenys:]
        obj = scipy.linalg.norm(L.T.dot(L) - M.numeric_instance(y)) + scipy.linalg.norm(A.dot(y)-b)
        print obj
    ipdb.set_trace()
    # need to know which matrix_mono is in each location

def sgd_solver(M, constraints, rank=2, maxiter=100, eta = 1):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    L = np.random.randn(rank, len(M.row_monos))
    L = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    y = np.random.randn(lenys,1)
    Bf = M.get_Bflat()
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)

    weightone = 1
    A = A*weightone; b = b*weightone
    
    consts = np.zeros((lenLs + lenys, 1))
    consts[-lenys:] = A.T.dot(b)
    coeffs = np.zeros((lenLs+lenys,lenLs+lenys))

    Ly = np.zeros((lenLs + lenys,1))
    Ly[0:lenLs] = L.T.flatten()[:,np.newaxis]
    Ly[lenLs:] = np.random.randn(lenys, 1)

    for i in xrange(maxiter):
        smallblock = L.dot(L.T)
        dy_L = -Bf.dot(np.kron(np.eye(rowsM), L.T))
        dy_y = A.T.dot(A) + Bf.dot(Bf.T)

        dL_L = np.kron(np.eye(rowsM), smallblock)
        dL_y = -dy_L.T

        coeffs[0:lenLs, 0:lenLs] = dL_L
        coeffs[0:lenLs, lenLs:] = dL_y

        coeffs[lenLs:, 0:lenLs] = dy_L
        coeffs[lenLs:, lenLs:] = dy_y
        
        grad = coeffs.dot(Ly) - consts;
        Ly = Ly - eta * grad
        L = Ly[0:lenLs].reshape((rowsM, rank)).T
        
        obj = scipy.linalg.norm(L.T.dot(L) - M.numeric_instance(y))**2 + scipy.linalg.norm(A.dot(y)-b)**2
        print obj
    ipdb.set_trace()
    # need to know which matrix_mono is in each location

def alternating_sgd_solver(M, constraints, rank=2, maxiter=100, tol=1e-3, eta = 1e-2):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    La = np.random.randn(rank, len(M.row_monos))
    #La = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2)
    
    coeffs = np.zeros((lenLs+lenys, lenLs+lenys))
    consts = np.zeros((lenLs+lenys, 1))
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)
    
    b = -A[:-1,0][:,np.newaxis]
    A = A[:-1,1:]
    
    counts = [len(M.term_to_indices_dict[yi]) for yi in M.matrix_monos[1:]]
    weight_fit = 1;
    for i in xrange(maxiter):
        # update y
        Q_y = A.T.dot(A) + weight_fit*np.diag(counts)
        currentM = La.T.dot(La)
        weights = [sum(currentM.flatten()[M.term_to_indices_dict[yi]]) for yi in M.matrix_monos[1:]]
        p_y = A.T.dot(b) + weight_fit*np.array(weights)[:,np.newaxis]
        y,_,_,_ = scipy.linalg.lstsq(Q_y, p_y)
        y_one = np.vstack((1,y))
        # print y, La.T.dot(La)
        # update L
        #ipdb.set_trace()
        Q_l = La.dot(La.T)
        p_l = La.dot(M.numeric_instance( y_one ))
        grad = Q_l.dot(La) - p_l
        #La,_,_,_ = scipy.linalg.lstsq(Q_l, p_l)
        La = La - eta*grad
        if i % 50 == 0:
            obj = scipy.linalg.norm(La.T.dot(La) - M.numeric_instance(y_one))**2 + scipy.linalg.norm(A.dot(y)-b)**2
            print i,obj

    return y,La

def alternating_solver(M, constraints, rank=2, maxiter=100, tol=1e-3, eta=0.001):
    lenys = len(M.matrix_monos)
    rowsM = len(M.row_monos)
    lenLs = rank*rowsM
    La = np.random.randn(rank, len(M.row_monos))
    #La = np.array([[1,1,1,1],[1,2,4,8]])/np.sqrt(2) + np.random.randn(rank, len(M.row_monos))*0.01;
    
    coeffs = np.zeros((lenLs+lenys, lenLs+lenys))
    consts = np.zeros((lenLs+lenys, 1))
    
    A,b = M.get_Ab(constraints, cvxoptmode = False)
    
    b = -A[:-1,0][:,np.newaxis]
    A = A[:-1,1:]
    
    counts = [len(M.term_to_indices_dict[yi]) for yi in M.matrix_monos[1:]]
    weight_fit = 1;
    for i in xrange(maxiter):
        # update y
        Q_y = A.T.dot(A) + weight_fit*np.diag(counts)
        currentM = La.T.dot(La)
        weights = [sum(currentM.flatten()[M.term_to_indices_dict[yi]]) for yi in M.matrix_monos[1:]]
        p_y = A.T.dot(b) + weight_fit*np.array(weights)[:,np.newaxis]
        if i==0:
            y,_,_,_ = scipy.linalg.lstsq(Q_y, p_y)
        
        y = y - eta * (Q_y.dot(y) - p_y)
        y_one = np.vstack((1,y))
        # print y, La.T.dot(La)
        # update L
        #ipdb.set_trace()
        Q_l = La.dot(La.T)
        p_l = La.dot(M.numeric_instance( y_one ))
        La,_,_,_ = scipy.linalg.lstsq(Q_l, p_l)
        My = M.numeric_instance( y_one )
        #ipdb.set_trace()
        
        U,D,V=scipy.linalg.svd(My)
        La = V[0:rank,:]*np.sqrt(D[0:rank, np.newaxis])
        if i % 50 == 0:
            obj = scipy.linalg.norm(La.T.dot(La) - M.numeric_instance(y_one))**2 + scipy.linalg.norm(A.dot(y)-b)**2
            print i,obj

    return y,La
    # need to know which matrix_mono is in each location
