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
    while m_new < m_old:
        m_old, g_old, h_old = m_new, g_new, h_new
        g_new = grad(u)
        if count % n == 0:
            h_new = -g_new
        else:
            h_new = -g_new + max(beta(g_old, g_new, h_old), 0) * h_old
        # Backtracking line search
        alpha = 1 / r * alpha_init
        m_x = 10e100
        u_x = u
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


def __main__():
    set_logging_level(log.DEBUG)
    # parameters for rosenbrock
    a = 1.0
    b = 10.0

    # input for rosenbrock
    x = np.array([-0.75, 0.7])
    # input for model4a
    u = np.array([
        0.8147,
        0.9058,
        0.1270,
        0.9134,
        0.6324,
        0.0975,
        0.2785,
        0.5469,
        0.9575,
        0.9649,
        0.1576,
        0.9706,
        0.9572,
        0.4854,
        0.8003,
        0.1419,
        0.4218,
        0.9157,
        0.7922,
        0.9595,
        0.6557,
        0.0357,
        0.8491,
        0.9340,
        0.6787,
        0.7577,
        0.7431,
        0.3922,
        0.6555,
        0.1712,
        0.7060,
        0.0318,
        0.2769,
        0.0462,
        0.0971,
        0.8235,
        0.6948,
        0.3171,
        0.9502,
        0.0344,
        0.4387,
        0.3816,
        0.7655,
        0.7952,
        0.1869,
        0.4898,
        0.4456,
        0.6463,
        0.7094,
        0.7547,
        0.2760,
        0.6797,
        0.6551,
        0.1626,
        0.1190,
        0.4984,
        0.9597,
        0.3404,
        0.5853,
        0.2238,
        0.7513,
        0.2551,
        0.5060,
        0.6991,
        0.8909,
        0.9593,
        0.5472,
        0.1386, 
        0.1493,
        0.2575,
        0.8407,
        0.2543,
        0.8143,
        0.2435,
        0.9293,
        0.3500,
        0.1966,
        0.2511,
        0.6160,
        0.4733])

    # input for model4a
    g = np.zeros(80)
    g[61] = 1.0
    g[78] = 1.0
    # input for model4a
    A = np.ones(148)
    
    # testing objective and gradients
    fn.rosenbrock(x,a, b)
    fn.rosenbrock_grad(x, a, b)
    fn.model4a_objective(u, A, g)
    fn.model4a_gradient(u, A, g)

    # Steepest Descent with backtracking
    f = lambda x: fn.rosenbrock(x, a, b)
    grad = lambda x: fn.rosenbrock_grad(x, a, b)
    hess = lambda x: fn.rosenbrock_hess(x, a, b)
    newtons_method(f, grad, hess, x, tol=1e-6)
    steepest_descent_backtracking(f, grad, x, alpha=1)
    
    # Conjugate Gradient of rosenbrock
    steepest_descent_conjugate(f, grad, x, alpha_init=1, n=20, beta=fn.beta_1)

    # Conjugate Gradient of model4a
    u = np.zeros(80)
    f = lambda u: fn.model4a_objective(u, A, g)
    grad = lambda u: fn.model4a_gradient(u, A, g)
    hess = lambda u: fn.model4a_hessian(u, A, g)
    print(hess(u))
    #steepest_descent_conjugate(f, grad, u, alpha_init=1, n=20, beta=fn.beta_1)

    newtons_method(f, grad, hess, u, tol=1e-6)


def __main__test():
        # input for model4a
    u = np.array([
        0.8147,
        0.9058,
        0.1270,
        0.9134,
        0.6324,
        0.0975,
        0.2785,
        0.5469,
        0.9575,
        0.9649,
        0.1576,
        0.9706,
        0.9572,
        0.4854,
        0.8003,
        0.1419,
        0.4218,
        0.9157,
        0.7922,
        0.9595,
        0.6557,
        0.0357,
        0.8491,
        0.9340,
        0.6787,
        0.7577,
        0.7431,
        0.3922,
        0.6555,
        0.1712,
        0.7060,
        0.0318,
        0.2769,
        0.0462,
        0.0971,
        0.8235,
        0.6948,
        0.3171,
        0.9502,
        0.0344,
        0.4387,
        0.3816,
        0.7655,
        0.7952,
        0.1869,
        0.4898,
        0.4456,
        0.6463,
        0.7094,
        0.7547,
        0.2760,
        0.6797,
        0.6551,
        0.1626,
        0.1190,
        0.4984,
        0.9597,
        0.3404,
        0.5853,
        0.2238,
        0.7513,
        0.2551,
        0.5060,
        0.6991,
        0.8909,
        0.9593,
        0.5472,
        0.1386, 
        0.1493,
        0.2575,
        0.8407,
        0.2543,
        0.8143,
        0.2435,
        0.9293,
        0.3500,
        0.1966,
        0.2511,
        0.6160,
        0.4733])
    
    f = np.array([
        0.8276,
        3.9456,
        -4.9622,
        4.8403,
        2.9894,
        -2.6788,
        -1.2564,
        2.0678,
        5.9835,
        6.4761,
        -2.9499,
        5.9816,
        4.0644,
        2.5033,
        2.7696,
        -0.9908,
        1.1392,
        5.1424,
        4.2488,
        4.7884,
        1.6421,
        -3.9092,
        1.8731,
        3.6070,
        0.3175,
        2.2978,
        1.5792,
        -1.3590,
        3.0317,
        -3.5712,
        -0.2933,
        -2.7038,
        -1.9176,
        -1.6771,
        -4.4043,
        3.6004,
        1.7816,
        -1.7667,
        -0.2797,
        -1.6397,
        -1.2903,
        0.2238,
        1.4346,
        0.5476,
        -3.5096,
        -1.6091,
        -0.1381,
        2.5677,
        0.7389,
        3.1101,
        -2.8609,
        3.3196,
        2.5639,
        -0.9866,
        -3.5648,
        1.1061,
        3.5476,
        0.3216,
        -0.5678,
        -0.4766,
        0.7133,
        -1.2582,
        -0.6056,
        -0.0006,
        1.5608,
        1.6810,
        1.1494,
        -0.9432,
        -2.2373,
        -1.2449,
        0.9007,
        -0.5076,
        0.6174,
        -0.8126,
        2.2608,
        0.2671,
        -3.1128,
        0.9078,
        -0.2301,
        0.4531])



    # input for model4a
    g = np.zeros(80)
    g[61] = 1.0
    g[78] = 1.0
    # input for model4a
    A = np.ones(148)
    
    x = fn.model4a_objective(u, A, g)
    y = fn.model4a_gradient(u, A, g)
    #print(x)
    print("\n")
    print(y.round(4)-f)
    hess = lambda u: fn.model4a_hessian(u, A, g)
    print(hess(u))
    
    
    

if __name__ == "__main__":
    __main__()