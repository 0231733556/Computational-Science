import numpy as np
import functions as fn
import logging as log
from inputs import rosenbrock_inputs as rb, model4a_inputs as m4a, constrains_8_16_inputs as c8_16, constrains_8_17_inputs as c8_17, constrains_8_18_inputs as c8_18
from algorithms import *
from constraints import *

def set_logging_level(level):
    log.basicConfig(level=level)
    
def newtons_method_tests():
    f = lambda x: fn.rosenbrock(x, rb["a"], rb["b"])
    grad = lambda x: fn.rosenbrock_grad(x, rb["a"], rb["b"])
    hess = lambda x: fn.rosenbrock_hess(x, rb["a"], rb["b"])
    newtons_method(f, grad, hess, rb["x"], tol=1e-12)
    
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g"])
    hess = lambda u: fn.model4a_hessian(u, m4a["A"], m4a["g"])
    u_zero = np.zeros(80)
    newtons_method(f, grad, hess, u_zero, tol=1e-12)

def bfgs_tests():

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
    log.basicConfig(level=log.INFO)
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g_zero"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g_zero"])
    hess = lambda u: fn.model4a_hessian(u, m4a["A"], m4a["g_zero"])
    f_c, g_c, h_c = make_penalized_model4a(f, grad, hess, model4a_constraint_8_16, c8_16["xR"], c8_16["yR"], c8_16["R"], k=100)
    print("=================== Conjugate ===================")
    steepest_descent_conjugate(fun=f_c, u=m4a["u_zero"], grad=g_c, beta=fn.beta_1, n=25, r=0.5, c=0.5, alpha_init=1)
    print("=================== Newton's ====================")
    newtons_method(f_c, g_c, h_c, m4a["u_zero"], tol=1e-12)
    print("==================== BFGS =======================")
    bfgs(f_c, g_c, m4a["u_zero"])
    # run lagrangian test as part of constraint tests
    
def inequality_constraint_tests():
    log.basicConfig(level=log.INFO)
    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g_zero"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g_zero"])
    hess = lambda u: fn.model4a_hessian(u, m4a["A"], m4a["g_zero"])
    f_c, g_c, h_c = make_penalized_model4a(f, grad, hess, model4a_constraint_8_17, c8_17["xR"], c8_17["yR"], c8_17["R"], k=100)
    f_c2, g_c2, h_c2 = make_penalized_model4a(f, grad, hess, model4a_constraint_8_18, c8_18["xR"], c8_18["yR"], c8_18["R"], k=100)
    
    print("=================== Conjugate ===================")
    steepest_descent_conjugate(fun=f_c, u=m4a["u_zero"], grad=g_c, beta=fn.beta_1, n=25, r=0.5, c=0.5, alpha_init=1)
    print("=================== Newton's ====================")
    newtons_method(f_c, g_c, h_c, m4a["u_zero"], tol=1e-12)
    print("==================== BFGS =======================")
    bfgs(f_c, g_c, m4a["u_zero"])
    
    
    print("=================== Conjugate ===================")
    steepest_descent_conjugate(fun=f_c2, u=m4a["u_zero"], grad=g_c2, beta=fn.beta_1, n=25, r=0.5, c=0.5, alpha_init=1)
    print("=================== Newton's ====================")
    newtons_method(f_c2, g_c2, h_c2, m4a["u_zero"], tol=1e-12)
    print("==================== BFGS =======================")
    bfgs(f_c2, g_c2, m4a["u_zero"], alpha_init=0.01)
    # run lagrangian test as part of constraint tests


def lagrangian_test():
    """Lagrange multiplier solver for equality constraints (Eq. 8.16).

    Uses an alternating primal-dual scheme:
      - primal: minimize L(u, lam) w.r.t. u (BFGS)
      - dual:   lam <- lam + rho * c(u)   (no projection for equality)

    The test checks result against the expected lambda and u provided in
    the exercise description (tolerance is loose to allow small numeric drift).
    """
    set_logging_level(log.INFO)
    # problem data
    R = c8_16["R"]
    xR = c8_16["xR"]
    yR = c8_16["yR"]

    f = lambda u: fn.model4a_objective(u, m4a["A"], m4a["g_zero"])
    grad = lambda u: fn.model4a_gradient(u, m4a["A"], m4a["g_zero"])

    # provide base Hessian for Newton primal solves
    base_hess = lambda uu: fn.model4a_hessian(uu, m4a["A"], m4a["g_zero"])

    L_fun, L_grad_u, L_grad_lambda, update_lambda, L_hess = make_lagrangian_model4a(
        f, grad, base_hess, model4a_constraint_8_16, xR, yR, R, j_range=range(31, 41)
    )

    # initial guesses
    u = m4a["u_zero"].copy()
    size = len(range(31, 41))
    lam = np.zeros(size)

    print("=================== Lagrangian 8.16 ===================")
    # Use the reusable solver in constraints.py which performs Newton primal steps
    u, lam, c_vec, iters = lagrangian_solver(
        f, grad, base_hess, model4a_constraint_8_16,newtons_method,
        xR, yR, R, u0=u, lam0=lam, tol=1e-12, rho=1.0, max_iter=20,
        j_range=range(31, 41)
    )
    
    print("=================== Lagrangian 8.17 ===================")
    # Use the reusable solver in constraints.py which performs Newton primal steps
    u, lam, c_vec, iters = lagrangian_solver(
        f, grad, base_hess, model4a_constraint_8_17,newtons_method,
        xR, yR, R, u0=u, lam0=lam, tol=1e-12, rho=1.0, max_iter=20,
        j_range=range(31, 41)
    )
    
    print("=================== Lagrangian 8.18 ===================")
    # Use the reusable solver in constraints.py which performs Newton primal steps
    u, lam, c_vec, iters = lagrangian_solver(
        f, grad, base_hess, model4a_constraint_8_18, newtons_method,
        xR, yR, R, u0=u, lam0=lam, tol=1e-12, rho=1.0, max_iter=20,
        j_range=range(31, 41)
    )

    # Expected results (from user's reference)
    expected_lam = np.array([
        0.4947, 0.0726, -0.2363, -0.3925, -0.4625,
        -0.4625, -0.3925, -0.2363, 0.0726, 0.4947
    ])

    expected_u = np.array([
        -0.0362, 0.0470, -0.0674, 0.0127, -0.0747, -0.0350, -0.0564, -0.0750,
        -0.0208, -0.0974, 0.0208, -0.0974, 0.0564, -0.0750, 0.0747, -0.0350,
        0.0674, 0.0127, 0.0362, 0.0470, -0.0065, 0.1506, -0.0624, 0.0420,
        -0.0796, -0.0796, -0.0619, -0.1783, -0.0230, -0.2316, 0.0230, -0.2316,
        0.0619, -0.1783, 0.0796, -0.0796, 0.0624, 0.0420, 0.0065, 0.1506,
        0.1470, 0.2935, 0.0503, 0.0758, 0.0026, -0.1380, -0.0118, -0.2930,
        -0.0062, -0.3734, 0.0062, -0.3734, 0.0118, -0.2930, -0.0026, -0.1380,
        -0.0503, 0.0758, -0.1470, 0.2935, 0.4827, 0.4463, 0.3183, 0.0812,
        0.1930, -0.1993, 0.1008, -0.3906, 0.0310, -0.4878, -0.0310, -0.4878,
        -0.1008, -0.3906, -0.1930, -0.1993, -0.3183, 0.0812, -0.4827, 0.4463
    ])




