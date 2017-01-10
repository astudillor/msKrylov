#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
from msKrylov import msGmres

A = np.random.rand(10, 10)
b = np.random.rand(10)
m_ = 6
V, H = msGmres(A, b, [1.9, 12.0, 3], m=m_)
print("|AV - VH|_2 = ", np.linalg.norm(A.dot(V[:, 0:m_]) - V.dot(H)))
