import numpy as np
import functions as fn
import logging as log
import sympy as sp
import inputs as inp
from algorithms import steepest_descent, steepest_descent_backtracking, steepest_descent_conjugate, newtons_method, set_logging_level

def __main__():
    set_logging_level(log.DEBUG)
    
    for iter in range(3):
        n = [3,30,300]
        u=np.zeros(n[iter])
        for i in range(n[iter]):
            u[i]=i+1
        f= lambda x: fn.ex1_objective(x)
        grad= lambda x: fn.ex1_gradient(x)
        steepest_descent_conjugate(f,grad,u,alpha_init=1.0,c=0.5,r=0.5,n=20,beta=fn.beta_1)
    
    
    

def test_suite():
    grad = lambda u: fn.model4a_gradient(u, inp.A, inp.g)
    should_be_zero = grad(inp.u).round(4) - inp.expected_grad
    assert np.allclose(should_be_zero, 0)

if __name__ == "__main__":
    test_suite()
    __main__()