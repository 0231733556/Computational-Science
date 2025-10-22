import numpy as np
import functions as fn
import logging as log
import sympy as sp

EPSILON = 1e-6

def set_logging_level(level):
    log.basicConfig(level=level)
    fn.set_logging_level(level)

def steepest_descent(fun, grad, u, alpha=0.04):
    """
    Performs Steepest Descent optimization to minimize given Function.
    Args:
        alpha (float): Step size for the descent.
        fun (callable): The objective function to minimize.
        grad (callable): Function to compute the gradient of the objective function.
    Returns:
        m_old, u - alpha * h (tuple): The minimum value of the objective function and the corresponding point.
    
    """
    m_new = fun(u)
    m_old = 10e100
    while m_new < m_old:
        m_old = m_new
        g = grad(u)
        h = -g
        u = u + alpha * h
        m_new = fun(u)
    
    return m_old, u - alpha * h

def steepest_descent_backtracking(fun, grad, u, alpha=0.04, c=0.5, r=0.8):
    """
    Performs Steepest Descent optimization to minimize given Function,
    using backtracking line search to determine step size.
    
    Args:
        alpha (float): Initial step size for the descent.
        fun (callable): The objective function to minimize.
        grad (callable): Function to compute the gradient of the objective function.
        c (float): Parameter for sufficient decrease condition (0 < c < 1).
        r (float): Step size reduction factor (0 < r < 1).
    Returns:
        m_old, u-alpha*h (tuple): The minimum value of the objective function and the corresponding point.
    """
    m_new = fun(u)
    m_old = 10e100
    count = 0
    while m_new < m_old:
        m_old = m_new
        g=grad(u)
        h=-g
        # Backtracking line search
        alpha=1/r*alpha
        m_x =10e100
        u_x = u
        while m_x > m_new +c*alpha*np.dot(h,g):
            alpha = r*alpha
            u_x = u + alpha*h
            m_x = fun(u_x)
        m_new = m_x
        u = u_x
        count += 1
    log.debug(f"{count} iterations; fun(u): {m_old}; u: {u-alpha*h}")
    return m_old, u-alpha*h
           

def steepest_descent_conjugate(fun, grad, u, alpha_init=0.04, c=0.5, r=0.8, n=100, beta=lambda f, g, h: 0):   
    """
    Minimizes a given function using the conjugate gradient method with backtracking line search.
    """
    m_new = fun(u)
    alpha = alpha_init
    m_old = 10e100
    g_new = 0
    h_new = 0
    count = 0
    # continue only up until the function stops improving
    while m_new < m_old:
        m_old, g_old, h_old = m_new, g_new, h_new
        g_new = grad(u)
        # resets steepest descent every steps to correct for conjugacy drift
        if count % n == 0:
            h_new = -g_new
        else:
            # finds a conjugate direction
            h_new = -g_new + max(beta(g_new, g_old, h_old), 0) * h_old
        # Backtracking line search
        alpha = 1 / r * alpha_init
        m_x = 10e100
        u_x = u
        # while it doesn't obey armijo condition aka sufficient descent
        while m_x > m_new + c * alpha * np.dot(h_new, g_new):
            alpha = r * alpha
            u_x = u + alpha * h_new
            m_x = fun(u_x)
        #if count % 1000 == 0:
            #log.debug(f"Iteration {count} ; fun(u) : {fun(u)}; u: {u}")
        m_new = m_x
        u = u_x
        count += 1
    #log.debug(f"m_old: {m_old}, m_new: {m_new}, ||g_new||: {np.linalg.norm(g_new)}, ||h_new||: {np.linalg.norm(h_new)}, alpha: {alpha}, count: {count}")
    log.info(f"Conjugate Gradient converged in {count} iterations.")
    log.debug(f"{count} iterations; fun(u): {m_old}; u: {u - alpha * h_new}")
    return m_old, u - alpha * h_new, count

def newtons_method(fun,grad,hess,u,tol):
    """
    Minimizes a given function using Newton's method.
    
    Args:
        a (float): Parameter for the objective function.
        g (np.ndarray): Gradient vector.
        u (np.ndarray): Initial guess for the minimum.
        tol (float): Tolerance for convergence.
    
    Returns:
        u*,m(u*) (tuple): The minimum value of the objective function and the corresponding point.
        """
    g = grad(u)
    count = 0
    while np.linalg.norm(g) > tol:
        K=hess(u)
        try:
            h = -np.linalg.solve(K, g)
        except np.linalg.LinAlgError:
            log.warning("Hessian is singular or not invertible. Aborting...")
            return None
        u=u+h
        g = grad(u)
        count += 1
        if count % 10 == 0:
            log.debug(f"Iteration {count} ; fun(u) : {fun(u)}; u: {u}")
    log.debug(f" {count} iterations; fun(u) : {fun(u)}; u: {u}")
    return u, fun(u)
