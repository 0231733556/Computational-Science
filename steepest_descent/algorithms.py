import numpy as np
import functions as fn
EPSILON = 1e-6

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
    #TODO: implement steepest descent method
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
    return m_old, u - alpha * h_new, count

def __main__():
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
    print(fn.rosenbrock(x,a, b))
    print(fn.rosenbrock2_grad(x, a, b))
    print(fn.model4a_objective(u, A, g))
    print(fn.model4a_gradient(u, A, g))

    # Steepest Descent with backtracking
    f = lambda x: fn.rosenbrock(x, a, b)
    grad = lambda x: fn.rosenbrock2_grad(x, a, b)
    print(steepest_descent_backtracking(f, grad, x, alpha=1))
    
    # Conjugate Gradient of rosenbrock
    print(steepest_descent_conjugate(f, grad, x, alpha_init=1, n=20, beta=fn.beta_1))

    # Conjugate Gradient of model4a
    u = np.zeros(80)
    f = lambda u: fn.model4a_objective(u, A, g)
    grad = lambda u: fn.model4a_gradient(u, A, g)
    print(steepest_descent_conjugate(f, grad, u, alpha_init=1, n=20, beta=fn.beta_1))



if __name__ == "__main__":
    __main__()