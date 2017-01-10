#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from scipy._lib.six import xrange
from scipy.linalg import get_blas_funcs
from scipy.sparse.linalg.isolve.utils import make_system

__all__ = ['msFom']


def msFom(A, b, shifts, m=200, x0=None, tol=1e-5, maxiter=None, M=None,
            callback=None):
    """
    Multi-Shift Gmres(m)
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b, xtype=None)
    n = len(b)
    if maxiter is None:
        maxiter = n * 10

    matvec = A.matvec
    psolve = M.matvec
    xtype = x.dtype

    # Initialization
    # BLAS functions
    axpy, dot, norm, rotmat = get_blas_funcs(
        ['axpy', 'dot', 'nrm2', 'rotg'], dtype=xtype)
    # Relative tolerance
    tolb = tol * norm(b)
    # Compute initial residual:
    r = b - matvec(x)
    rnrm2 = norm(r)
    nshifts = len(shifts)
    X = np.zeros((n, nshifts), dtype=xtype)
    for j in xrange(0, nshifts):
        X[:, j] = np.array(x)
    errors = np.zeros(nshifts, dtype=xtype)
    V = np.zeros((n, m + 1), dtype=xtype)
    H = np.zeros((m + 1, m), dtype=xtype)
    # Compute residuals
    converged = True
    for j in xrange(0, nshifts):
        rj = r + shifts[j]*X[:, j]
        errors[j] = norm(rj)
        converged = converged and (errors[j] < tolb)
    if callback is not None:
        callback(errors)
    if converged:
        return V, H

    # Begin iteration
    # for _iter in range(0, maxit):
    V[:, 0] = r / rnrm2
    for i in xrange(0, m):
        # Construct orthonormal basis using Gram-Schmidt
        w = psolve(matvec(V[:, i]))
        for k in range(0, i + 1):
            H[k, i] = dot(V[:, k], w)
            axpy(V[:, k], w, None, -H[k, i])
        H[i + 1, i] = norm(w)
        V[:, i + 1] = w / H[i + 1, i]
    # Multi-Shift part
    for j in xrange(0, nshifts):
        # Hj = H - w_j I
        Hj = np.array(H)
        for k in xrange(0, m):
            Hj[k, k] -= shifts[j]
        # s = |r_j| e_1
        s = np.zeros(m+1, dtype=xtype)
        s[0] = errors[j]
        # Apply Givens rotation to H_j[:m, :m] and s
        # Solve Triangular system with scipy.linalg.solve_triangular
        # Update X
        # Residuals? ¯\_(ツ)_/¯
    return V, H
