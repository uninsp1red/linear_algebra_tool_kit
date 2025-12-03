import numpy as np
import math 


EPSILON = 10e-12
# Convergence/pivot tolerance used across algorithms.

def generate_matrix(n: int):
    """Return an n x n matrix with random ints in [-100, 100) as float64."""
    test_matrix = np.random.randint(-100,100,size=(n,n))
  
    return test_matrix.astype(np.float64)

def generate_normal_matrix(n: int):
    """Generate a symmetric positive semi-definite matrix via A.T @ A."""
    A = generate_matrix(n)

    return A.T @ A

def diagonally_dominant_matrix(n):
    """Construct a random diagonally dominant matrix of size n x n."""
    A = np.random.rand(n, n)
    
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + np.random.randint(1, 10)
    
    return A


def generate_vector(A):
    """Create a random integer vector sized to match matrix A."""
    n = A.shape[0]
    test_vector = np.random.randint(-100,100, n)
    return test_vector

def LU_decomposition(A: np.ndarray):
    """Perform LU decomposition with full pivoting.

    Returns permutation matrices P, Q, combined LU matrix, swap count, and rank.
    """
    LU = A.copy().astype(np.float64)
    n = A.shape[0]
    P = np.eye(n)
    Q = np.eye(n)
    swaps = 0
    rank = n
    for i in range(n):
        pivot = 0
        max_index_row = max_index_col = i

        # Search for the largest pivot in the remaining submatrix.
        for j in range(i, n):
            for k in range(i, n):
                if abs(pivot) < abs(LU[j,k]):
                    pivot = LU[j,k]
                    max_index_row = j
                    max_index_col = k
        # Reorder rows/columns to move the pivot to the diagonal.
        if max_index_row != i:
            LU[[i,max_index_row],:] = LU[[max_index_row,i], :]
            P[[i,max_index_row],:] = P[[max_index_row,i], :]
            swaps += 1
        if max_index_col != i:
            LU[:, [i, max_index_col]] = LU[:, [max_index_col, i]]
            Q[:, [i, max_index_col]] = Q[:, [max_index_col, i]]
            swaps+=1
        if abs(LU[i,i]) < EPSILON:
            rank = i
            break
        for j in range(i+1, n):
            l_elem = LU[j,i]/LU[i,i]
            LU[j, i:] = LU[j, i:] - LU[i, i:] * l_elem
            LU[j,i] = l_elem

    return P, Q, LU, swaps, rank

def get_L(LU: np.ndarray):
    """Extract the lower-triangular matrix (with unit diagonal) from LU."""
    n = LU.shape[0]
    L = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            L[i, j] = LU[i, j]
    return L

def get_U(LU: np.ndarray):
    """Extract the upper-triangular matrix from LU."""
    n = LU.shape[0]
    U = np.eye(n)
    for i in range(n):
        for j in range(i, n):
            U[i,j] = LU[i,j]

    return U

def determinant(A: np.ndarray):
    """Calculate the determinant using LU decomposition."""
    _, _, LU, swaps, rank = LU_decomposition(A)
    n = LU.shape[0]
    if rank < n: 
        return 0
    det = 1
    for i in range(n):
        det *=  LU[i,i]

    return (-1)**(swaps) * det

def slae_sol(A: np.ndarray, b: np.array):
    """Solve Ax = b using LU decomposition with full pivoting."""
    P, Q, LU, _, rank = LU_decomposition(A)
    n = LU.shape[0]
    b_conv = P@b
    y = np.zeros(n)
    z = np.zeros(n)
    y[0] = b_conv[0]
    for i in range(1,n):
        y[i] = b_conv[i] - np.dot(LU[i,:(i)], y[:(i)])

    if is_compatible(LU, y) == False:
        raise ValueError("uncompatible slae")

    for i in range(rank-1, -1, -1):
        z[i] = (y[i] - np.dot(z[(i+1):], LU[i, (i+1):])) / LU[i,i]

    return Q@z

def is_compatible(LU: np.ndarray, y: np.array):
    """Check if the system is consistent for the current LU and RHS vector."""
    n = LU.shape[0]
    have_not_zero_val = True
    for i in range(n - 1, -1, -1):
        if have_not_zero_val == False:
            return False
        if y[i] == 0:
            continue
        for j in range(n - 1, i - 1, -1):
            if abs(LU[i,j]) > EPSILON:
                have_not_zero_val = True
            else:
                have_not_zero_val = False
    return True

def invert_matrix(A: np.ndarray):
    """Compute the inverse of A via LU decomposition."""
    P, Q, LU, _, rank = LU_decomposition(A)
    n = LU.shape[0]
    if rank != n:
        raise ValueError("degenerate matrix")
    Y = np.zeros((n,n))
    for i in range(n):
        Y[0, i] = P[0, i]
        for j in range(1, n):
            Y[j, i] = P[j, i] - np.dot(LU[j,:j], Y[:j, i])

    inverted_A = np.zeros((n,n))
    for i in range(n):
        inverted_A[-1, i] = Y[-1, i] / LU[-1,-1]
        for j in range(n-2, -1, -1):
            inverted_A[j, i] = (Y[j, i] - np.dot(inverted_A[(j+1):, i], LU[j, (j+1):])) / LU[j,j]

    return Q@inverted_A

def find_conditionality(A: np.ndarray, norm: str = 'norm'):
    """Return the condition number of A for the given norm identifier."""
    A_inv = invert_matrix(A)
    try:
        fn = _norm_handlers[norm]
    except:
        raise ValueError("unknown norm")

    norm_A, norm_A_inv = fn(A), fn(A_inv)

    return norm_A * norm_A_inv


def _norm(A: np.ndarray):
    """Column sum norm (1-norm)."""
    n = A.shape[0]
    max_num = -10**(9)
    for i in range(n):
        max_in_col = 0
        for j in range(n):
            max_in_col += abs(A[j, i])
        if max_num < max_in_col:
            max_num = max_in_col
    
    return max_num

def _inf_norm(A: np.ndarray):
    """Row sum norm (infinity norm)."""
    n = A.shape[0]
    max_num = -10**(9)
    for i in range(n):
        max_in_col = 0
        for j in range(n):
            max_in_col += abs(A[i, j])
        if max_num < max_in_col:
            max_num = max_in_col
    
    return max_num

def _frobenius_norm(A: np.ndarray):
    """Frobenius norm."""
    n = A.shape[0]
    frob_norm = 0
    for i in range(n):
        for j in range(n):
            frob_norm += (A[i,j])**2
    
    return math.sqrt(frob_norm)

_norm_handlers = {
    'norm': _norm,
    'inf': _inf_norm,
    'frobenius': _frobenius_norm,
}

def QR_decomposition(A: np.ndarray):
    """Compute the QR decomposition using Householder reflections."""
    R = A.copy()
    n = A.shape[0]
    Q = np.eye(n)
    for i in range(n - 1):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = 1
        u_i = x - np.linalg.norm(x) * e
        B = np.zeros([n,n])
        B[i:, i:] = np.outer(u_i,u_i)
        P_i = np.eye(n) - 2 * (B)/np.linalg.norm(u_i)**2
        R = P_i @ R
        Q = Q @ P_i
    return Q, R

def QR_slae_solve(A: np.ndarray, b: np.array):
    """Solve Ax = b via QR decomposition and back substitution."""
    n = A.shape[0]
    Q, R = QR_decomposition(A)
    y = Q.T @ b
    x = np.zeros(n)
    x[-1] = y[-1] / R[-1,-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(x[i+1:], R[i, i+1:]))/R[i,i]

    return x

def Jacobi(A: np.ndarray, b: np.array, a_posteriori_estimate: bool = False):
    """Iterative Jacobi solver; optionally returns iterations taken."""
    n = A.shape[0]
    D = np.zeros_like(A)
    np.fill_diagonal(D, np.diag(A))
    x_prev = np.zeros(n)
    x_next = np.diag(1 / np.diag(D)) @ b

    D_inv = np.diag(1 / np.diag(D))
    B = D_inv @ (D - A)
    q = _norm(B)
    k = 0

    for iteration in range(100):
        x_prev = x_next.copy()
        x_next = np.zeros(n)
        for i in range(n):
            A_temp = _ignore_index(A[i,:], i)
            x_temp = _ignore_index(x_prev, i)
            x_next[i] = (b[i] - A_temp@x_temp)/A[i,i]
        
        if np.linalg.norm(x_next - x_prev) < EPSILON*(1-q)/q:
            k = iteration
            break

    if a_posteriori_estimate:
        return x_next, k
    
    return x_next

def Seidel(A: np.ndarray, b: np.array, a_posteriori_estimate: bool = False):
    """Gauss-Seidel iterative solver; optionally returns iterations taken."""
    n = A.shape[0]
    D = np.zeros_like(A)
    np.fill_diagonal(D, np.diag(A))
    x_next = np.diag(1 / np.diag(D)) @ b

    L = np.tril(A, -1) 
    R = np.triu(A, 1)
    L_plus_R = L + R
    B = -np.linalg.inv(D) @ L_plus_R
    B_L = np.tril(B, -1) 
    B_R = np.triu(B, 1)
    norm_B_R = _norm(B_R)
    norm_B_L = _norm(B_L)
    k = 0

    for iteration in range(100000):
        x_prev = x_next.copy()
        for i in range(n):
            A_temp = _ignore_index(A[i,:], i)
            x_temp = _ignore_index(x_next, i)
            x_next[i] = (b[i] - A_temp@x_temp)/A[i,i]
        
        if np.linalg.norm(x_next - x_prev) < EPSILON*(1-norm_B_L)/norm_B_R:
            k = iteration
            break

    if a_posteriori_estimate:
        return x_next, k
    
    return x_next

def a_priori_estimate(A: np.ndarray, b: np.array):
    """Estimate iterations needed for Jacobi method given tolerance."""
    B = find_B_from_Jacobi(A)
    q = _norm(B)
    D = np.diag(np.diag(A))        
    D_inv = np.diag(1 / np.diag(A))
    c = np.linalg.norm(D_inv @ b)
    k = 0
    comparable = EPSILON * (1 - q) / c

    return int(math.log(comparable, q)) + 1

def find_B_from_Jacobi(A):
    """Build the Jacobi iteration matrix B = -D^-1(A - D)."""
    n = A.shape[0]
    D = np.diag(np.diag(A))        
    D_inv = np.diag(1 / np.diag(A)) 
    return -D_inv @ (A - D)

def _ignore_index(a: np.array, i: int):
    """Return a copy of array `a` without the element at index i."""
    mask = np.ones(a.shape, dtype=bool)
    mask[i] = False
    return a[mask]