import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import minimize
import scipy.linalg
import numpy as np
import scipy.linalg
from scipy.optimize import differential_evolution, basinhopping, shgo

def solve_linear_system(A, b, regularization=1e-6, method="auto"):
    """
    Solve Ax = b in a numerically stable manner.
    
    Parameters:
        A (ndarray): Coefficient matrix (m x n).
        b (ndarray): Right-hand side vector (m,).
        regularization (float): Small regularization term for ill-conditioned problems.
        method (str): Solver method, "auto", "direct", "qr", or "svd".

    Returns:
        x (ndarray): Solution vector (n,).
    """
    m, n = A.shape

    # 1. If A is a square matrix (m=n), use direct solving method first
    if m == n:
        try:
            # Check if A is singular
            if np.linalg.cond(A) < 1 / np.finfo(A.dtype).eps:
                return scipy.linalg.solve(A, b, assume_a="gen")
            else:
                # If A is ill-conditioned, use regularization
                A_reg = A + regularization * np.eye(n)
                return scipy.linalg.solve(A_reg, b, assume_a="pos")
        except np.linalg.LinAlgError:
            pass  # Direct solving failed, try other methods

    # 2. If A is a non-square matrix (overdetermined/underdetermined), use least squares
    if m != n or method in ["qr", "svd"]:
        if method == "qr" or (method == "auto" and m >= n):
            # QR decomposition (for overdetermined systems)
            Q, R = np.linalg.qr(A)
            return scipy.linalg.solve(R, Q.T @ b)
        elif method == "svd" or method == "auto":
            # SVD decomposition (for ill-conditioned matrices)
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            S_inv = np.diag(1 / (S + regularization))  # Inverse singular value matrix (with regularization)
            return Vt.T @ S_inv @ U.T @ b

    # 3. Default least squares solving (most robust)
    return np.linalg.lstsq(A, b, rcond=None)[0]

def validate_inputs(x, Omegas, d):
    """Ensure that d is an integer >= 1 and x, Omegas have the same length."""
    if not isinstance(d, int) or d < 1:
        raise ValueError("d must be an integer greater than or equal to 1")
    
    x = np.asarray(x)
    Omegas = np.asarray(Omegas)
    
    if x.shape != Omegas.shape:
        raise ValueError("x and Omegas must have the same length")

    return x, Omegas

def EPSR_odd(x, Omegas, d): 
    """Extended parameter shift rule for odd-order derivatives (e.g., d=1,3,5...)."""
    x, Omegas = validate_inputs(x, Omegas, d)

    p = (Omegas ** d) * (-1 if d % 4 == 3 else 1)
    Ao = np.sin(np.outer(x, Omegas))
    
    # return np.linalg.solve(Ao.T, p)
    return solve_linear_system(Ao.T, p)

def EPSR_even(x, Omegas, d): 
    """Extended parameter shift rule for even-order derivatives (e.g., d=2,4,6...)."""
    q = (Omegas ** d) * (-1 if d % 4 == 2 else 1)
    q = np.insert(q, 0, 0)  # Ensure q[0] = 0
    Omegas = np.insert(Omegas, 0, 0)  # Ensure Omegas[0] = 0

    x, Omegas = validate_inputs(x, Omegas, d)

    Ae = np.cos(np.outer(x, Omegas))
    
    # return np.linalg.solve(Ae.T, q)
    return solve_linear_system(Ae.T, q)

def EPSR(x, Omegas, d):
    if d % 2 == 1:
        return EPSR_odd(x, Omegas, d)
    else:
        return EPSR_even(x, Omegas, d)
    