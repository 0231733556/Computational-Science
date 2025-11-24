import numpy as np
import functions as fn
import logging as log
import sympy as sp
from inputs import rosenbrock_inputs as rb, model4a_inputs as m4a
from algorithms import *
from constraints import *
from tests import *

def __main__():

    # testing objective and gradients
    fn.rosenbrock(rb["x"], rb["a"], rb["b"])
    fn.rosenbrock_grad(rb["x"], rb["a"], rb["b"])
    fn.model4a_objective(m4a["u"], m4a["A"], m4a["g"])
    fn.model4a_gradient(m4a["u"], m4a["A"], m4a["g"])

    # Steepest Descent with backtracking
    f = lambda x: fn.rosenbrock(x, rb["a"], rb["b"])
    grad = lambda x: fn.rosenbrock_grad(x, rb["a"], rb["b"])
    hess = lambda x: fn.rosenbrock_hess(x, rb["a"], rb["b"])
    newtons_method(f, grad, hess, rb["x"], tol=1e-6)
    steepest_descent_backtracking(f, grad, rb["x"], alpha=1)

    # Conjugate Gradient of rosenbrock
    steepest_descent_conjugate(f, grad, rb["x"], alpha_init=1, n=20, beta=fn.beta_1)

    # Conjugate Gradient of model4a
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g"])
    print(grad(m4a["u"]).round(4) - m4a["expected_grad"])
    hess = lambda u: fn.model4a_hessian(u, m4a["A"], m4a["g"])
    print(hess(m4a["u"]))
    #steepest_descent_conjugate(f, grad, u, alpha_init=1, n=20, beta=fn.beta_1)

    u = np.zeros(80)
    newtons_method(f, grad, hess, u, tol=1e-6)

    bfgs(f, grad, m4a["u"])

if __name__ == "__main__":
    set_logging_level(log.INFO)
    constraint_tests()
    #__main__()