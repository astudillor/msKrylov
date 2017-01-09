from __future__ import division, print_function

import numpy as np
from scipy._lib.six import xrange
from scipy.linalg import get_blas_funcs
from scipy.sparse.linalg.isolve.utils import make_system

__all__ = ['msGmres']

def msGmres(A, b, shifts, m=200, x0=None, tol=1e-5, maxiter=None, M=None, 
            callback=None):
    """
    Multi-Shift Gmres(m)
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b, xtype = None)
    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    xtype = x.dtype

    axpy, dot, norm = get_blas_funcs(['axpy', 'dot', 'nrm2'], dtype=xtype)

    # Relative tolerance
    tolb = tol*norm(b)
    # Compute initial residual:
    r = b - matvec(x)
    rnrm2 = norm(r)

    # We should call the callback func

    # Initialization
    V = np.zeros((n, m+1), dtype=xtype)
    H = np.zeros((m+1, m), dtype=xtype)
    cs = np.zeros(m+1, dtype=xtype)
    sn = np.zeros(m+1, dtype=xtype)
    e1 = np.zeros(m+1, dtype=xtype)
    e1[0] = 1.0

    # Begin iteration
    #for _iter in range(0, maxit):
    V[:,0] = r/rnrm2
    s = rnrm2*e1
    for i in range(0, m):
        # Construct orthonormal basis using Gram-Schmidt
        w = psolve(matvec(V[:, i]))
        for k in range(0, i+1):
            H[k, i] = dot(V[:, k], w)
            #w = w - H[k, i]*V[:, k]
            axpy(V[:, k], w, None, -H[k, i])
        H[i+1, i] = norm(w)
        V[:, i+1] = w/H[i+1, i]
	# Multi-Shift with \bar{H} 
	# We should call the callback func
    return V, H
