import sympy as sp
import numpy as np
from sympy import Matrix
from sympy.printing.pycode import pycode

def grad_solver(expr, vars):
    grad = sp.Matrix([sp.diff(expr, var) for var in vars])
    grad = sp.simplify(grad)
    sp.pprint(grad)

def hess_solver(expr, vars):
    hess = sp.hessian(expr, vars)
    hess = sp.simplify(hess)
    sp.pprint(hess)

# variables
x1, x2, a, b = sp.symbols('x1 x2 a b')
expr = (x1 - a)**2 + b*(x2 - x1**2)**2
expr = expr.subs({a:1, b:10}) 

grad_solver(expr, (x1, x2))
print("\n") 
hess_solver(expr, (x1, x2))