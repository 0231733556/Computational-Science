import sympy as sp
import numpy as np
from sympy import Matrix
from sympy.printing.pycode import pycode
from math import sqrt
from sympy.printing.pretty import pretty

SQRT2 = sp.sqrt(2)

def grad_solver(expr, vars, file_name = None):
    grad = sp.Matrix([sp.diff(expr, var) for var in vars])
    grad = sp.simplify(grad)
    if file_name:
        s = pretty(grad, use_unicode=True, wrap_line=False)   
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        sp.pprint(grad)

def hess_solver(expr, vars, file_name = None):
    hess = sp.hessian(expr, vars)
    hess = sp.simplify(hess)
    if file_name:
        s = pretty(hess, use_unicode=True, wrap_line=False)   
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        sp.pprint(hess)

# variables
x1, x2, a, b = sp.symbols('x1 x2 a b')
expr = (x1 - a)**2 + b*(x2 - x1**2)**2
#expr = expr.subs({a:1, b:10}) 


def build_model4a_expr():
    # variables
    N = 80
    u = sp.symbols(f'u1:{N+1}')      # u1..u80
    g = sp.symbols(f'g1:{N+1}')      # g1..g80
    A = sp.symbols('a1:149')         # a1..a148

    # helper: 1-based MATLAB index -> symbol
    def U(k): return u[k-1]

    # helper term: w * (sqrt(a^2 + b^2) - c)^2
    def term(a, b, c, w):
        return w * (sp.sqrt(a**2 + b**2) - c)**2

    # objective
    m = -sp.Add(*[g[i]*u[i] for i in range(N)])  # -g^T u

    # 1) i=1..10
    for i in range(1, 11):
        a = U(2*i-1)
        b = 1 + U(2*i)
        m += term(a, b, 1, A[i-1])

    # 2) i=1..9
    for i in range(1, 10):
        a = 1 + U(2*i+1)
        b = 1 + U(2*i+2)
        m += term(a, b, SQRT2, A[10 + (i-1)])

    # 3) i=1..9
    for i in range(1, 10):
        a = 1 - U(2*i-1)
        b = 1 + U(2*i)
        m += term(a, b, SQRT2, A[19 + (i-1)])

    # 4) i=1..30
    for i in range(1, 31):
        a = 1 + U(2*i+20) - U(2*i)
        b =     U(2*i+19) - U(2*i-1)
        m += term(a, b, 1, A[28 + (i-1)])

    # 5) i=1..36, t = 2*floor((i-1)/9)
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        a = 1 + U(2*i+1+t) - U(2*i-1+t)
        b =     U(2*i+2+t) - U(2*i  +t)
        m += term(a, b, 1, A[58 + (i-1)])

    # 6) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        a = 1 + U(2*i+21+t) - U(2*i-1+t)
        b = 1 + U(2*i+22+t) - U(2*i  +t)
        m += term(a, b, SQRT2, A[94 + (i-1)])

    # 7) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        a = 1 - U(2*i+19+t) + U(2*i+1+t)
        b = 1 + U(2*i+20+t) - U(2*i+2+t)
        m += term(a, b, SQRT2, A[121 + (i-1)])

    return m, u, A, g

def term_expr():
    """
    Return a symbolic term and the symbolic input vector for the expression
    w * (sqrt(a(u)^2 + b(u)^2) - c)^2.

    This function constructs symbolic variables using sympy and builds the term
    m = A[0] * (sqrt(a**2 + b**2) - c)**2 where
        a = a_vec . u + alpha
        b = b_vec . u + beta

    Notes
    - The function creates symbol arrays for u, A (weights), a_vec, b_vec and the
      scalars c, alpha, beta. In the current implementation N is set to 1, so
      u, A, a_vec and b_vec contain a single element each.
    - The returned expression m is a sympy expression; u is the tuple of sympy
      symbols representing the input vector components.
    - The implementation expects sympy to be available as the name `sp` and numpy
      as `np` in the module namespace.

    Returns
    - m (sympy.Expr): symbolic expression for the term w*(sqrt(a(u)^2 + b(u)^2)-c)^2
    - u (tuple(sympy.Symbol)): tuple of sympy symbols for the components of u
    """
    N = 1
    u = sp.symbols(f'u1:{N+1}')      
    A = sp.symbols(f'w1:{N+1}')
    c = sp.symbols('c')
    a_vec = sp.symbols(f'a_coeff1:{N+1}')
    alpha = sp.symbols('alpha')
    b_vec = sp.symbols(f'b_vec1:{N+1}')
    beta = sp.symbols('beta')
    a = np.sum([a_vec[i] * u[i] for i in range(N)]) + alpha
    b = np.sum([b_vec[i] * u[i] for i in range(N)]) + beta

    m = A[0] * (sp.sqrt(a**2 + b**2) - c)**2
    return m, u

def expr3(u:np.array):
    u[0]=sp.symbols('u1')
    u[1]=sp.symbols('u2')
    u[2]=sp.symbols('u3')   
    u[3]=sp.symbols('u4')
    u[4]=sp.symbols('u5')
    f=(u[0])**2 * (u[1])**2 * (u[2])**2 * (u[3])**2 * (u[4])**2
    return f, u

def expr2(u: np.ndarray) -> float:
    """
    Objective function:
        f(u) = -u_n + (u_1)^4 + sum_{j=2}^{n} (u_j - u_{j-1})^4

    Args:
        u (np.ndarray): 1D array of variables [u1, u2, ..., un]

    Returns:
        float: Objective value
    """
    u = np.asarray(u, dtype=float)
    n = u.size
    if n < 1:
        raise ValueError("Input vector u must have at least one element.")
    N=1
    u=sp.symbols(f'u1:{N+1}')
    term1 = u[0] ** 4
    diffs = np.diff(u)
    term2 = np.sum(diffs ** 4)
    term3 = -u[-1]
    f=term1 + term2 + term3
    return f, u
#m, u, A, g = build_model4a_expr()
k=[0,0,0,0,0]
f, u = expr3(k)

iter =2
n = [3,30,300]
u=np.zeros(n[iter])
for i in range(n[iter]):
    u[i]=i+1
f, u = expr2(u)
print(f)
print("\n")
grad_solver(f, u, file_name="printout_grad.txt")
print("\n")
hess_solver(f, u, file_name="printout_hess.txt")
