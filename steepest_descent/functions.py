import numpy as np
import logging as log

def set_logging_level(level):
    log.basicConfig(level=level)

SQRT2 = np.sqrt(2.0)
# Rosenbrock
def rosenbrock(x, a, b):
    temp = (x[0] - a)**2 + b*(x[1]-x[0]**2)**2
      
    log.debug(f"Evaluating rosenbrock at x={x}")
    log.debug(f"rosenbrock value: {temp}")
    return temp

def rosenbrock_grad(x, a, b):
    x1, x2 = x
    d1 = 2*(x1 - a) - 4*b*x1*(x2 - x1**2)
    d2 = 2*b*(x2 - x1**2)
    return np.array([d1, d2])

def rosenbrock_hess(x, a, b):
    x1, x2 = x
    h11 = 2 - 4*b*x2 + 12*b*x1**2
    h12 = -4*b*x1
    h21 = h12
    h22 = 2*b
    return np.array([[h11, h12], [h21, h22]])

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

def model4a_hessian(u, A, g, eps=1e-12):
    u = np.asarray(u, float)
    A = np.asarray(A, float)
    H = np.zeros((u.size, u.size), dtype=float)

    def _term_values(u, a_idx, a_coeff, a_const, b_idx, b_coeff, b_const):
        # a(u) = a_const + sum_j a_coeff[j] * u[a_idx[j]]
        a = a_const + np.dot(a_coeff, u[a_idx]) if len(a_idx) else a_const
        b = b_const + np.dot(b_coeff, u[b_idx]) if len(b_idx) else b_const
        r = np.hypot(a, b)
        return a, b, r


    def _add_hess(u, H, w, c, a_idx, a_coeff, a_const, b_idx, b_coeff, b_const, eps=1e-12):
        # Exact Hessian for w*(||[a,b]|| - c)^2 with affine a(u), b(u)
        a, b, r = _term_values(u, a_idx, a_coeff, a_const, b_idx, b_coeff, b_const)
        if r < eps:
            return
        # Build small vectors A and B over the involved indices S = a_idx ∪ b_idx
        S = list(dict.fromkeys(list(a_idx) + list(b_idx)))  # unique, keep order
        m = len(S)
        A = np.zeros(m); B = np.zeros(m)
        for j, idx in enumerate(a_idx):
            A[S.index(idx)] += a_coeff[j]
        for j, idx in enumerate(b_idx):
            B[S.index(idx)] += b_coeff[j]

        va = a * A
        vb = b * B
        g = (va + vb) / r                              # ∇r restricted to S
        # ∇^2 r on S: (AA^T + BB^T)/r - (va+vb)(va+vb)^T / r^3
        H_r = (np.outer(A, A) + np.outer(B, B)) / r - np.outer(va + vb, va + vb) / (r**3)
        K_S = 2.0 * w * (np.outer(g, g) + (r - c) * H_r)

        # scatter-add into full H
        for p, ip in enumerate(S):
            for q, iq in enumerate(S):
                H[ip, iq] += K_S[p, q]

    # block 1
    for i in range(1, 11):
        _add_hess(u, H, A[i-1], 1.0,
                  [2*i-2], [1.0], 0.0,
                  [2*i-1], [1.0], 1.0)

    # block 2
    for i in range(1, 10):
        _add_hess(u, H, A[10+i-1], SQRT2,
                  [2*i],   [1.0], 1.0,
                  [2*i+1], [1.0], 1.0)

    # block 3
    for i in range(1, 10):
        _add_hess(u, H, A[19+i-1], SQRT2,
                  [2*i-2], [-1.0], 1.0,
                  [2*i-1], [ 1.0], 1.0)

    # block 4
    for i in range(1, 31):
        _add_hess(u, H, A[28+i-1], 1.0,
                  [2*i+18, 2*i-1], [1.0, -1.0], 1.0,
                  [2*i+17, 2*i-2], [1.0, -1.0], 0.0)

    # block 5
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        _add_hess(u, H, A[58+i-1], 1.0,
                  [2*i-1+t, 2*i-3+t], [1.0, -1.0], 1.0,
                  [2*i+t,   2*i-2+t], [1.0, -1.0], 0.0)

    # block 6
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        _add_hess(u, H, A[94+i-1], SQRT2,
                  [2*i+19+t, 2*i-3+t], [1.0, -1.0], 1.0,
                  [2*i+20+t, 2*i-2+t], [1.0, -1.0], 1.0)

    # block 7
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        _add_hess(u, H, A[121+i-1], SQRT2,
                  [2*i-1+t, 2*i+17+t], [1.0, -1.0], 1.0,
                  [2*i+t,   2*i+18+t], [-1.0, 1.0], 1.0)

    return H   

def beta_1(f_new,f_old,h_old):
    """
    Compute the first beta coefficient,
    by dividing the squared norm of the new gradient,
    by the squared norm of the old gradient.
    
    (f_new)**2 / (f_old)**2
    
    Args:
        f_new (np.ndarray): The new gradient vector.
        f_old (np.ndarray): The old gradient vector.
    Returns:
        float: The computed beta_1 value.
    """
    return np.dot(f_new,f_new)/np.dot(f_old,f_old)

def beta_2(f_new,f_old,h_old):
    """
    Compute the second beta coefficient,
    by dividing the dot product of the new gradient
    and the difference between the new and old gradients,
    by the squared norm of the old gradient.
    
    (f_new * (f_new - f_old)) / (f_old)**2
    
    Args:
        f_new (np.ndarray): The new gradient vector.
        f_old (np.ndarray): The old gradient vector.
    Returns:
        float: The computed beta_2 value.
    """
    return np.dot(f_new,f_new-f_old)/np.dot(f_old,f_old)

def beta_3(f_new,f_old,h_old):
    """
    Compute the third beta coefficient,
    by dividing the dot product of the new gradient
    and the difference between the new and old gradients,
    by the dot product of the old step direction
    and the difference between the new and old gradients.

    f_new * (f_new - f_old) / (h_old * (f_new - f_old))

    Args:
        f_new (np.ndarray): The new gradient vector.
        f_old (np.ndarray): The old gradient vector.
        h_old (np.ndarray): The old step direction.
    Returns:
        float: The computed beta_3 value.
    """
    return np.dot(f_new,f_new-f_old)/np.dot(h_old, (f_new-f_old))

def beta_4(f_new,f_old,h_old):
    """
    Compute the fourth beta coefficient,
    by dividing the squared norm of the new gradient vector,
    by the dot product of the old step direction
    and the difference between the new and old gradients.

    f_new**2 / (h_old * (f_new - f_old))

    Args:
        f_new (np.ndarray): The new gradient vector.
        f_old (np.ndarray): The old gradient vector.
        h_old (np.ndarray): The old step direction.
    Returns:
        float: The computed beta_4 value.
    """
    return np.dot(f_new,f_new)/np.dot(h_old, (f_new-f_old))


