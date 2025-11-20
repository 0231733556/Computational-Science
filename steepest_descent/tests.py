import numpy as np
import functions as fn
import logging as log
from inputs import rosenbrock_inputs as rb, model4a_inputs as m4a
from algorithms import *
from constraints import *


def bfgs_tests():
    set_logging_level(log.DEBUG)
    
    
    f = lambda x: fn.rosenbrock(x, rb["a"], rb["b"])
    grad = lambda x: fn.rosenbrock_grad(x, rb["a"], rb["b"])
    hess = lambda x: fn.rosenbrock_hess(x, rb["a"], rb["b"])
    bfgs(f, grad, rb["x"])
    
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g"])
    hess = lambda u: fn.model4a_hessian(u, m4a["A"], m4a["g"])
    u_zero = np.zeros(80)
    bfgs(f, grad, u_zero)

def m4a_gradient_test():
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g"])
    should_be_zero = grad(m4a["u"]).round(4) - m4a["expected_grad"]
    assert np.allclose(should_be_zero, 0)
    
def l_bfgs_tests():
    set_logging_level(log.DEBUG)
    
    f = lambda x: fn.rosenbrock(x, rb["a"], rb["b"])
    grad = lambda x: fn.rosenbrock_grad(x, rb["a"], rb["b"])
    hess = lambda x: fn.rosenbrock_hess(x, rb["a"], rb["b"])
    l_bfgs(f, grad, rb["x"])
    
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g"])
    hess = lambda u: fn.model4a_hessian(u, m4a["A"], m4a["g"])
    u_zero = np.zeros(80)
    l_bfgs(f, grad, u_zero, n_li=10)
    
def constraint_tests():
    set_logging_level(log.DEBUG)
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g_zero"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g_zero"])
    
    R=9
    yR=12.5
    xR=4.5
    f_c, g_c = make_penalized_model4a(f,grad, xR, yR, R, k=100)
    
    u_zero = np.zeros(80)
    bfgs(f_c, g_c, u_zero)