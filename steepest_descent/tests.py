import numpy as np
import functions as fn
import logging as log
from inputs import rosenbrock_inputs as rb, model4a_inputs as m4a, constrains_8_16_inputs as c8_16, constrains_8_17_inputs as c8_17, constrains_8_18_inputs as c8_18
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
    f_c, g_c = make_penalized_model4a(f, grad, model4a_constraint_8_18, c8_18["xR"], c8_18["yR"], c8_18["R"], k=100)
    steepest_descent_conjugate(fun=f_c, u=m4a["u_zero"], grad=g_c, beta=fn.beta_1, n=25, r=0.5, c=0.5, alpha_init=1)
    #bfgs(f_c, g_c, m4a["u_zero"])