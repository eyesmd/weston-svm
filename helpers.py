import math
import numpy as np
from numpy import array, dot
import cvxopt
from cvxopt import matrix

# CVXOPT wrapper
# Credit: https://scaron.info/blog/quadratic-programming-in-python.html

# solves:
#   minimize    0.5 x^t P x + q^t x
#   subject to  G x <= h
#               A x = b

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = P.astype(float)
    q = q.astype(float)
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        G = G.astype(float)
        h = h.astype(float)
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            A = A.astype(float)
            b = b.astype(float)
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return array(sol['x']).reshape((P.shape[1],))

def run_cvxopt_test():
    # CVXOPT test
    M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = dot(M.T, M)
    q = -dot(M.T, array([3., 2., 3.]))
    G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = array([3., 2., -2.]).reshape((3,))
    
    cvxopt.solvers.options['show_progress'] = False
    arr = cvxopt_solve_qp(P, q, G, h)
    cvxopt.solvers.options['show_progress'] = True

    EPS = 1e-3
    assert math.isclose(arr[0], 0.12997347, rel_tol=EPS)
    assert math.isclose(arr[1], -0.06498674, rel_tol=EPS)
    assert math.isclose(arr[2], 1.74005305, rel_tol=EPS)

run_cvxopt_test()
