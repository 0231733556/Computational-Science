import numpy as np 

SQRT2 = np.sqrt(2.0)
EPSILON = 1e-6
a = 1.0
b = 10.0

# Rosenbrock
def rosenbrock(x, a, b):
    return (x[0] - a)**2 + b*(x[1]-x[0]**2)**2

def rosenbrock2_grad(x, a, b):
    x1, x2 = x
    d1 = 2*(x1 - a) - 4*b*x1*(x2 - x1**2)
    d2 = 2*b*(x2 - x1**2)
    return np.array([d1, d2])

def model4a_objective(u, A, g):
    u = np.asarray(u, float)
    A = np.asarray(A, float)
    g = np.asarray(g, float)

    def U(k):  # 1-based MATLAB index -> 0-based Python
        return u[k-1]

    m = -np.dot(g, u)

    # 1) i=1..10
    for i in range(1, 11):
        a = U(2*i-1)
        b = 1.0 + U(2*i)
        r = np.hypot(a, b)
        m += A[i-1] * (r - 1.0)**2

    # 2) i=1..9
    for i in range(1, 10):
        a = 1.0 + U(2*i+1)
        b = 1.0 + U(2*i+2)
        r = np.hypot(a, b)
        m += A[10 + (i-1)] * (r - SQRT2)**2

    # 3) i=1..9
    for i in range(1, 10):
        a = 1.0 - U(2*i-1)
        b = 1.0 + U(2*i)
        r = np.hypot(a, b)
        m += A[19 + (i-1)] * (r - SQRT2)**2

    # 4) i=1..30
    for i in range(1, 31):
        a = 1.0 + U(2*i+20) - U(2*i)
        b =        U(2*i+19) - U(2*i-1)
        r = np.hypot(a, b)
        m += A[28 + (i-1)] * (r - 1.0)**2

    # 5) i=1..36   (block offset t = 2*floor((i-1)/9))
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        a = 1.0 + U(2*i+1+t) - U(2*i-1+t)
        b =        U(2*i+2+t) - U(2*i  +t)
        r = np.hypot(a, b)
        m += A[58 + (i-1)] * (r - 1.0)**2

    # 6) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        a = 1.0 + U(2*i+21+t) - U(2*i-1+t)
        b = 1.0 + U(2*i+22+t) - U(2*i  +t)
        r = np.hypot(a, b)
        m += A[94 + (i-1)] * (r - SQRT2)**2

    # 7) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        a = 1.0 - U(2*i+19+t) + U(2*i+1+t)
        b = 1.0 + U(2*i+20+t) - U(2*i+2+t)
        r = np.hypot(a, b)
        m += A[121 + (i-1)] * (r - SQRT2)**2

    return m


def model4a_gradient(u, A, g, eps=1e-12):
    """
    Hard-coded analytic gradient of model4a_objective. Returns shape (80,).
    """
    u = np.asarray(u, float)
    A = np.asarray(A, float)
    g = np.asarray(g, float)
    N = u.size
    grad = -g.copy()

    def add_term(coeffs_a, coeffs_b, c, w):
        """
        Term: w * (sqrt(a^2 + b^2) - c)^2
        a = const + sum_j coeffs_a[j] * u[j]
        b = const + sum_j coeffs_b[j] * u[j]
        coeffs_* is list of tuples (index, coefficient, constant_contrib)
        Where constant_contrib is added only once to build a and b.
        """
        # Build a, b; each tuple is (idx, coeff)
        a_const = 0.0
        b_const = 0.0
        a = 0.0
        b = 0.0
        for idx, coeff in coeffs_a:
            if idx is None:  # use None to encode constants
                a_const += coeff
            else:
                a += coeff * u[idx]
        for idx, coeff in coeffs_b:
            if idx is None:
                b_const += coeff
            else:
                b += coeff * u[idx]
        a += a_const
        b += b_const

        r = np.hypot(a, b)
        if r < eps:
            return  # flat; contribution is zero in the limit (rare here)
        common = w * 2.0 * (r - c) / r
        # ∂/∂u_j = common * (a*∂a/∂u_j + b*∂b/∂u_j)
        for idx, coeff in coeffs_a:
            if idx is not None:
                grad[idx] += common * a * coeff
        for idx, coeff in coeffs_b:
            if idx is not None:
                grad[idx] += common * b * coeff

    # Helper to convert 1-based MATLAB k -> 0-based index
    def I(k): return k-1

    # 1) i=1..10
    for i in range(1, 11):
        w = A[i-1]; c = 1.0
        add_term([(I(2*i-1), 1.0)],
                 [(None, 1.0), (I(2*i), 1.0)],
                 c, w)

    # 2) i=1..9
    for i in range(1, 10):
        w = A[10 + (i-1)]; c = SQRT2
        add_term([(None, 1.0), (I(2*i+1), 1.0)],
                 [(None, 1.0), (I(2*i+2), 1.0)],
                 c, w)

    # 3) i=1..9
    for i in range(1, 10):
        w = A[19 + (i-1)]; c = SQRT2
        add_term([(None, 1.0), (I(2*i-1), -1.0)],
                 [(None, 1.0), (I(2*i),   1.0)],
                 c, w)

    # 4) i=1..30
    for i in range(1, 31):
        w = A[28 + (i-1)]; c = 1.0
        add_term([(None, 1.0), (I(2*i+20), 1.0), (I(2*i), -1.0)],
                 [(I(2*i+19), 1.0), (I(2*i-1), -1.0)],
                 c, w)

    # 5) i=1..36
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        w = A[58 + (i-1)]; c = 1.0
        add_term([(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(I(2*i+2+t), 1.0), (I(2*i+t), -1.0)],
                 c, w)

    # 6) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        w = A[94 + (i-1)]; c = SQRT2
        add_term([(None, 1.0), (I(2*i+21+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(None, 1.0), (I(2*i+22+t), 1.0), (I(2*i+t), -1.0)],
                 c, w)

    # 7) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        w = A[121 + (i-1)]; c = SQRT2
        add_term([(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i+19+t), -1.0)],
                 [(None, 1.0), (I(2*i+20+t), 1.0), (I(2*i+2+t), -1.0)],
                 c, w)

    return grad

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
    while abs(m_new - m_old) > EPSILON:
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
    while abs(m_new - m_old) > EPSILON:
        m_old = m_new
        g=grad(u)
        h=-g
        # Backtracking line search
        alpha=1/r*alpha
        m_x =10e100
        u_x = u
        while abs(m_x -m_new +c*alpha*np.dot(g,h))>EPSILON:
            alpha = r*alpha
            u_x = u + alpha*h
            m_x = fun(u_x)
        m_new = m_x, u = u_x
    return m_old, u-alpha*h
           


def __main__():
    
    x = np.array([-0.75, 0.7])
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

    g = np.array([0.0]*80)
    g[61] = 1.0
    g[78] = 1.0

    A = np.ones(148)
    print(rosenbrock(x,a, b))
    print(rosenbrock2_grad(x, a, b))
    print(model4a_objective(u, A, g))
    print(model4a_gradient(u, A, g))
    
    f = lambda x: rosenbrock(x, a, b)
    g = lambda x: rosenbrock2_grad(x, a, b)
    print(steepest_descent_backtracking(f, g, x, alpha=0.001))
    


if __name__ == "__main__":
    __main__()