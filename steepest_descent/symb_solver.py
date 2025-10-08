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




m, u, A, g = build_model4a_expr()

print(m)
print("\n")
#grad_solver(m, u)
print("\n") 
#hess_solver(m, u, file_name="hess_pretty.txt")
