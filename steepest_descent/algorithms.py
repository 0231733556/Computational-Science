import numpy as np
import functions as fn
import logging as log

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


def _approx_inv_hessian(L_old,diff_f,diff_u):
    """
    Compute an approximate inverse Hessian matrix using the BFGS update formula.

    Args:
        L_old (np.ndarray): The old inverse Hessian matrix.
        diff_f (np.ndarray): The difference in gradient vectors.
        diff_u (np.ndarray): The difference in parameter vectors.
    Returns:
        np.ndarray: The updated inverse Hessian matrix.
    """
    diff_u_dot_diff_f = np.inner(diff_u, diff_f) # scalar
    L_dot_diff_f = np.dot(L_old, diff_f) # vector
    
    part1_scalar = (diff_u_dot_diff_f + np.dot(diff_f, L_dot_diff_f)) / (np.inner(diff_u, diff_f)**2) # scalar
    part1_mat = np.outer(diff_u, diff_u) # matrix
    part_1 = part1_scalar * part1_mat # matrix
    
    part2_mat1 = np.outer(L_dot_diff_f, diff_u) # matrix
    part2_mat2 = np.dot(np.outer(diff_u, diff_f), L_old) # matrix
    part_2 = - (part2_mat1 + part2_mat2) /  diff_u_dot_diff_f
    L_new  = L_old + part_1 + part_2 # matrix
    return L_new

def bfgs_line_search(u, fun, grad, h, alpha_init, c1, c2, r, tol, m_new, g_new):
    # Determine initial search domain, but stop if acceptable stepsize is found
    
    # Initializing return values to avoid errors
    signal1, ux, m2, m3, g2, g3 = 0, None, None, None, None, None
    
    alpha3 = alpha_init
    ux = u + alpha3 * h
    m3 = fun(ux)
    g3 = grad(ux)
    if m3 <= m_new + c1 * alpha3 * np.dot(h, g_new) and np.dot(h, g3) >= c2 * np.dot(h, g_new):
        signal1 = 1
    while m3 < m_new + c1 * alpha3 * np.dot(h, g_new) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = u + alpha3 * h
        m3 = fun(ux)
        g3 = grad(ux)
        if m3 <= m_new + c1 * alpha3 * np.dot(h, g_new) and np.dot(h, g3) >= c2 * np.dot(h, g_new):
            signal1 = 1
    # Apply bisection method if no acceptable stepsize is found yet
    if signal1 == 0:
        signal2 = 0
        alpha1 = 0
        alpha2 = alpha3 * 0.5
        ux = u + alpha2 * h
        m2 = fun(ux)
        g2 = grad(ux)
        while signal2 == 0:
            if alpha3 - alpha1 < tol:
                signal2 = 1
                m2 = m_new
                g2 = g_new
            elif m2 > m_new + c1 * alpha2 * np.dot(h, g_new):
                alpha3 = alpha2
                m3 = m2
                g3 = g2
                alpha2 = 0.5 * (alpha1 + alpha2)
                ux = u + alpha2 * h
                m2 = fun(ux)
                g2 = grad(ux)
            elif np.dot(h, g2) < c2 * np.dot(h, g_new):
                alpha1 = alpha2
                alpha2 = 0.5 * (alpha2 + alpha3)
                ux = u + alpha2 * h
                m2 = fun(ux)
                g2 = grad(ux)
            else:
                signal2 = 1
    return signal1, ux, m2, m3, g2, g3


def bfgs(fun, grad, u, alpha_init=1, c1=1e-4, c2=0.9, r=0.5, tol=1e-15):
    """
    Minimizes a given function using the BFGS quasi-Newton method.
    
    Args:
        fun (callable): The objective function to minimize.
        grad (callable): Function to compute the gradient of the objective function.
        u (np.ndarray): Initial guess for the minimum.
        alpha_init (float): Initial step size for the line search.
        c1 (float): Parameter for the sufficient decrease condition (0 < c1 < 1).
        c2 (float): Parameter for the curvature condition (0 < c2 < 1).
        r (float): Step size reduction factor (0 < r < 1).
        tol (float): Tolerance for convergence.
    
    Returns:
        u*,m(u*) (tuple): The minimum value of the objective function and the corresponding point.
    """
    assert 0 < c1 < 1 and 0 < c2 < 1 and c1 < c2
    assert 0 < r < 1
    
    u = np.asarray(u, float)
    n = u.shape[0]
    Hinv_new = np.eye(n)  # Initial Hessian approximation
    m_new = fun(u)
    g2 = grad(u)
    m_old = 10e100
    g_new = 0
    count = 0
    diff_u = 0
    while m_new < m_old:
        m_old, g_old, Hinv_old = m_new, g_new, Hinv_new 
        g_new = g2
        diff_g = g_new - g_old
        # determine search direction
        if count == 0:
            h = -np.dot(Hinv_old, g_new)
        else:
            Hinv_new = _approx_inv_hessian(Hinv_old, diff_g, diff_u)
            h = -np.dot(Hinv_new, g_new)
        # Line search
        signal1, ux, m2, m3, g2, g3  = bfgs_line_search(u, fun, grad, h, alpha_init, c1, c2, r, tol, m_new, g_new)
            
        diff_u = ux - u
        u = ux
        count += 1
        if signal1 == 1:
            m_new = m3
            g2 = g3
        else:
            m_new = m2
        
        if count % 100 == 0:
            log.debug(f"Iteration {count} ; fun(u) : {m_old}; u: {u - diff_u}")
    
    log.debug(f" {count} iterations; fun(u) : {m_old}; u: {u - diff_u}")
    return m_old, u - diff_u


def l_bfgs(fun, grad, u, alpha_init=1, c1=1e-4, c2=0.9, r=0.5, tol=1e-15):
    """
    Minimizes a given function using the L-BFGS quasi-Newton method.
    
    Args:
        fun (callable): The objective function to minimize.
        grad (callable): Function to compute the gradient of the objective function.
        u (np.ndarray): Initial guess for the minimum.
        alpha_init (float): Initial step size for the line search.
        c1 (float): Parameter for the sufficient decrease condition (0 < c1 < 1).
        c2 (float): Parameter for the curvature condition (0 < c2 < 1).
        r (float): Step size reduction factor (0 < r < 1).
        tol (float): Tolerance for convergence.
    
    Returns:
        u*,m(u*) (tuple): The minimum value of the objective function and the corresponding point.
    """
    assert 0 < c1 < 1 and 0 < c2 < 1 and c1 < c2
    assert 0 < r < 1
    
    u = np.asarray(u, float)
    n = u.shape[0]
    Hinv_new = np.eye(n)  # Initial Hessian approximation
    m_new = fun(u)
    g2 = grad(u)
    m_old = 10e100
    g_new = 0
    count = 0
    diff_u = 0
    while m_new < m_old:
        m_old, g_old, Hinv_old = m_new, g_new, Hinv_new 
        g_new = g2
        diff_g = g_new - g_old
        # determine search direction
        if count == 0:
            h = -np.dot(Hinv_old, g_new)
        else:
            Hinv_new = _approx_inv_hessian(Hinv_old, diff_g, diff_u)
            h = -np.dot(Hinv_new, g_new)
        # Line search
        signal1, ux, m2, m3, g2, g3  = bfgs_line_search(u, fun, grad, h, alpha_init, c1, c2, r, tol, m_new, g_new)
            
        diff_u = ux - u
        u = ux
        count += 1
        if signal1 == 1:
            m_new = m3
            g2 = g3
        else:
            m_new = m2
        
        if count % 100 == 0:
            log.debug(f"Iteration {count} ; fun(u) : {m_old}; u: {u - diff_u}")
    
    log.debug(f" {count} iterations; fun(u) : {m_old}; u: {u - diff_u}")
    return m_old, u - diff_u


