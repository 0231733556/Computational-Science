import numpy as np
import functions as fn
import logging as log
import sympy as sp
import inputs as inp
from algorithms import *

def __main__():
    set_logging_level(log.DEBUG)
 
    # testing objective and gradients
    fn.rosenbrock(inp.x, inp.a, inp.b)
    fn.rosenbrock_grad(inp.x, inp.a, inp.b)
    fn.model4a_objective(inp.u, inp.A, inp.g)
    fn.model4a_gradient(inp.u, inp.A, inp.g)

    # Steepest Descent with backtracking
    f = lambda x: fn.rosenbrock(x, inp.a, inp.b)
    grad = lambda x: fn.rosenbrock_grad(x, inp.a, inp.b)
    hess = lambda x: fn.rosenbrock_hess(x, inp.a, inp.b)
    newtons_method(f, grad, hess, inp.x, tol=1e-6)
    steepest_descent_backtracking(f, grad, inp.x, alpha=1)

    # Conjugate Gradient of rosenbrock
    steepest_descent_conjugate(f, grad, inp.x, alpha_init=1, n=20, beta=fn.beta_1)

    # Conjugate Gradient of model4a
    f = lambda u: fn.model4a_objective(u, inp.A, inp.g)
    grad = lambda u: fn.model4a_gradient(u, inp.A, inp.g)
    print(grad(inp.u).round(4) - inp.expected_grad)
    hess = lambda u: fn.model4a_hessian(u, inp.A, inp.g)
    print(hess(inp.u))
    #steepest_descent_conjugate(f, grad, u, alpha_init=1, n=20, beta=fn.beta_1)

    u = np.zeros(80)
    newtons_method(f, grad, hess, u, tol=1e-6)

    bfgs(f, grad, inp.u)
    
def bfgs_tests():
    set_logging_level(log.DEBUG)
    
    
    f = lambda x: fn.rosenbrock(x, inp.a, inp.b)
    grad = lambda x: fn.rosenbrock_grad(x, inp.a, inp.b)
    hess = lambda x: fn.rosenbrock_hess(x, inp.a, inp.b)
    bfgs(f, grad, inp.x)
    
    f = lambda u: fn.model4a_objective(u, inp.A, inp.g)
    grad = lambda u: fn.model4a_gradient(u, inp.A, inp.g)
    hess = lambda u: fn.model4a_hessian(u, inp.A, inp.g)
    u_zero = np.zeros(80)
    bfgs(f, grad, u_zero)

def test_suite():
    grad = lambda u: fn.model4a_gradient(u, inp.A, inp.g)
    should_be_zero = grad(inp.u).round(4) - inp.expected_grad
    assert np.allclose(should_be_zero, 0)

if __name__ == "__main__":
    bfgs_tests()
    #__main__()