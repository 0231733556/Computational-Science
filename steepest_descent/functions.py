import numpy as np
import logging as log

def I(k): return k-1

def set_logging_level(level):
    log.basicConfig(level=level)

SQRT2 = np.sqrt(2.0)
# Rosenbrock
def rosenbrock(x, a, b):
    temp = (x[0] - a)**2 + b*(x[1]-x[0]**2)**2
      
    return temp.astype(float)

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

def _term_values(u: np.ndarray,
                coeffs_a: np.ndarray,
                coeffs_b: np.ndarray)-> tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Compute coefficient vectors and scalar term values from sparse coefficient specifications.

        This function constructs two dense coefficient vectors a_vec and b_vec (length N) from
        iterables of (index, coefficient) pairs, computes the linear combinations a = a_vec·u + a_const
        and b = b_vec·u + b_const, and returns those vectors together with the scalars a, b and r,
        where r = hypot(a, b) = sqrt(a**2 + b**2).

        Parameters
        ----------
        u : numpy.ndarray
            1-D array of length N representing the current variable vector u.
        coeffs_a : Iterable[tuple[int|None, float]]
            Iterable of (index, coefficient) pairs describing the contribution to a(u).
            - If index is an int, the coefficient is added to a_vec[index].
            - If index is None, the coefficient is treated as a constant term added to a_const.
        coeffs_b : Iterable[tuple[int|None, float]]
            Iterable of (index, coefficient) pairs describing the contribution to b(u),
            using the same convention as coeffs_a.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, float, float, float]
            A 5-tuple containing:
            - a_vec: numpy.ndarray of shape (N,) with accumulated coefficients for a(u).
            - b_vec: numpy.ndarray of shape (N,) with accumulated coefficients for b(u).
            - a: float, the scalar value a = a_vec.dot(u) + a_const.
            - b: float, the scalar value b = b_vec.dot(u) + b_const.
            - r: float, the Euclidean norm hypot(a, b) = sqrt(a**2 + b**2).

        Notes
        -----
        - The function expects that a working value N (the length of the coefficient vectors)
          is available in the surrounding scope or that len(u) matches the intended size.
        - It is the caller's responsibility to ensure that any integer indices provided in
          coeffs_a and coeffs_b are within the valid range [0, N-1]. Out-of-range indices will
          raise an IndexError when applied to the underlying arrays.
        """
        # a(u) = a_const + sum_j a_coeff[j] * u[a_idx[j]]
        N = u.size
        a_vec = np.zeros(N, dtype=float)
        b_vec = np.zeros(N, dtype=float)
        a_const = 0.0
        b_const = 0.0
        # Building A, B
        for idx, coeff in coeffs_a:
            if idx is not None:
                a_vec[idx] += coeff
            else:
                a_const += coeff
        for idx, coeff in coeffs_b:
            if idx is not None:
                b_vec[idx] += coeff
            else:
                b_const += coeff

        # Computing a[u], b[u] and r
        a = np.dot(a_vec, u) + a_const
        b = np.dot(b_vec, u) + b_const
        r = np.hypot(a, b)
        return a_vec, b_vec, a, b, r
    
    
def _add_obj(u: np.ndarray,
              coeffs_a: np.ndarray,
              coeffs_b: np.ndarray,
              c: float,
              w: float,
              m: np.ndarray) -> None:
    """
    Term: w * (sqrt(a[u]^2 + b[u]^2) - c)^2
    a[u] = a_vec . u + a_const
    b[u] = b_vec . u + b_const
    coeffs_* is list of tuples (index, coefficient, constant_contrib)
    Where constant_contrib is added only once to build a and b.
    
    Args:
        coeffs_a (list of (int or None, float)): Coefficients for a
        coeffs_b (list of (int or None, float)): Coefficients for b
        c (float): Constant to subtract from sqrt(a^2 + b^2)
        w (float): Weight of the term
    Returns:
        None: Updates total 
    """
    _, _, _, _, r = _term_values(u, coeffs_a, coeffs_b)


    # Adding component derivatives to total 
    np.add(m, w * (r - c) ** 2, out=m)

def _add_grad(u: np.ndarray,
              coeffs_a: np.ndarray,
              coeffs_b: np.ndarray,
              c: float,
              w: float,
              grad: np.ndarray,
              eps=1e-12) -> None:
    """
    Term: w * (sqrt(a[u]^2 + b[u]^2) - c)^2
    a[u] = a_vec . u + a_const
    b[u] = b_vec . u + b_const
    coeffs_* is list of tuples (index, coefficient, constant_contrib)
    Where constant_contrib is added only once to build a and b.
    
    Args:
        coeffs_a (list of (int or None, float)): Coefficients for a
        coeffs_b (list of (int or None, float)): Coefficients for b
        c (float): Constant to subtract from sqrt(a^2 + b^2)
        w (float): Weight of the term
    Returns:
        None: Updates grad in place
    """
    a_vec, b_vec, a, b, r = _term_values(u, coeffs_a, coeffs_b)

    if r < eps:
        return  # flat; contribution is zero in the limit (rare here)
    common = w * 2.0 * (r - c) / r

    # Adding component derivatives to total grad
    np.add(grad, common * (a * a_vec + b * b_vec), out=grad)

def _add_hess(u: np.ndarray,
              coeffs_a: np.ndarray,
              coeffs_b: np.ndarray,
              c: float,
              w: float,
              H: np.ndarray)-> None:
    """ 
    Add the Hessian contribution of a term
    
    Args:
        u (vector): the input for the hessian
        coeffs_a (list of (int or None, float)): Coefficients for a
        coeffs_b (list of (int or None, float)): Coefficients for b
        c (float): Constant to subtract from sqrt(a^2 + b^2)
        w (float): Weight of the term
        H (matrix): the hessian matrix to which we add our component's hessian (adding to the total)
        r_min (float): Minimum r to avoid division by zero
    Returns:
        None: Updates H in place
    """
    a_vec, b_vec, a, b, r = _term_values(u, coeffs_a, coeffs_b)

    # Clamp r away from zero
    #r = max(r, r_min)
    inv_r  = 1.0 / r
    inv_r3 = inv_r * inv_r * inv_r   # avoid r**3

    lin_combo = a * a_vec + b * b_vec

    # ∇r on S
    hess = (np.outer(a_vec, a_vec) + np.outer(b_vec, b_vec)) * (1 - c * inv_r)
    hess += np.outer(lin_combo, lin_combo) * c * inv_r3

    np.add(H, 2.0 * w * hess, out=H)

def model4a_objective(u, A, g):
    u = np.asarray(u, float)
    A = np.asarray(A, float)
    g = np.asarray(g, float)

    m = np.array(-np.dot(g, u), dtype=float)

    # 1) i=1..10: a = u_{2i-1}, b = 1 + u_{2i}, c = 1
    for i in range(1, 11):
        _add_obj(u,
                 [(I(2*i-1), 1.0)],
                 [(None, 1.0), (I(2*i), 1.0)],
                 c=1.0, w=A[i-1], m=m)

    # 2) i=1..9: a = 1 + u_{2i+1}, b = 1 + u_{2i+2}, c = sqrt(2)
    for i in range(1, 10):
        _add_obj(u,
                 [(None, 1.0), (I(2*i+1), 1.0)],
                 [(None, 1.0), (I(2*i+2), 1.0)],
                 c=SQRT2, w=A[10 + (i-1)], m=m)

    # 3) i=1..9: a = 1 - u_{2i-1}, b = 1 + u_{2i}, c = sqrt(2)
    for i in range(1, 10):
        _add_obj(u,
                 [(None, 1.0), (I(2*i-1), -1.0)],
                 [(None, 1.0), (I(2*i),   1.0)],
                 c=SQRT2, w=A[19 + (i-1)], m=m)

    # 4) i=1..30: a = 1 + u_{2i+20} - u_{2i}, b = u_{2i+19} - u_{2i-1}, c = 1
    for i in range(1, 31):
        _add_obj(u,
                 [(None, 1.0), (I(2*i+20), 1.0), (I(2*i), -1.0)],
                 [(I(2*i+19), 1.0), (I(2*i-1), -1.0)],
                 c=1.0, w=A[28 + (i-1)], m=m)

    # 5) i=1..36, t = 2*floor((i-1)/9):
    #    a = 1 + u_{2i+1+t} - u_{2i-1+t},  b = u_{2i+2+t} - u_{2i+t},  c = 1
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        _add_obj(u,
                 [(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(I(2*i+2+t), 1.0), (I(2*i+t), -1.0)],
                 c=1.0, w=A[58 + (i-1)], m=m)

    # 6) i=1..27, t = 2*floor((i-1)/9):
    #    a = 1 + u_{2i+21+t} - u_{2i-1+t},
    #    b = 1 + u_{2i+22+t} - u_{2i+t},  c = sqrt(2)
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        _add_obj(u,
                 [(None, 1.0), (I(2*i+21+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(None, 1.0), (I(2*i+22+t), 1.0), (I(2*i+t), -1.0)],
                 c=SQRT2, w=A[94 + (i-1)], m=m)

    # 7) i=1..27, t = 2*floor((i-1)/9):
    #    a = 1 - u_{2i+19+t} + u_{2i+1+t},
    #    b = 1 + u_{2i+20+t} - u_{2i+2+t},  c = sqrt(2)
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        _add_obj(u,
                 [(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i+19+t), -1.0)],
                 [(None, 1.0), (I(2*i+20+t), 1.0), (I(2*i+2+t), -1.0)],
                 c=SQRT2, w=A[121 + (i-1)], m=m)

    
    return m.astype(float)


def model4a_gradient(u, A, g) -> np.ndarray:
    """
    Compute the gradient of the model4a objective function at point u.
    Args:
        u (np.ndarray): The point at which to evaluate the gradient.
        A (np.ndarray): The weights for the terms in the objective.
        g (np.ndarray): The linear coefficients in the objective.
        eps (float): Small value to avoid division by zero.
    Returns:
        np.ndarray: The gradient vector at point u.
    """
    u = np.asarray(u, float)
    A = np.asarray(A, float)
    g = np.asarray(g, float)
    # derivative of first sum component
    grad = -g.copy()
    

    #pass (None,CONSTANT) to indicate constant term
    #pass (INDEX,COEFFICIENT) to indicate variable term, use sign of coefficient for +/- in term
    #pass terms sequewntially to build a and b, first a then b in expressiion sqrt(a^2 + b^2) using [] of tuples
    
    # 1) i=1..10
    for i in range(1, 11):
        w = A[i-1]; c = 1.0
        _add_grad(u, [(I(2*i-1), 1.0)],
                 [(None, 1.0), (I(2*i), 1.0)],
                 c, w, grad)

    # 2) i=1..9
    for i in range(1, 10):
        w = A[10 + (i-1)]; c = SQRT2
        _add_grad(u, [(None, 1.0), (I(2*i+1), 1.0)],
                 [(None, 1.0), (I(2*i+2), 1.0)],
                 c, w, grad)

    # 3) i=1..9
    for i in range(1, 10):
        w = A[19 + (i-1)]; c = SQRT2
        _add_grad(u, [(None, 1.0), (I(2*i-1), -1.0)],
                 [(None, 1.0), (I(2*i),   1.0)],
                 c, w, grad)

    # 4) i=1..30
    for i in range(1, 31):
        w = A[28 + (i-1)]; c = 1.0
        _add_grad(u, [(None, 1.0), (I(2*i+20), 1.0), (I(2*i), -1.0)],
                 [(I(2*i+19), 1.0), (I(2*i-1), -1.0)],
                 c, w, grad)

    # 5) i=1..36
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        w = A[58 + (i-1)]; c = 1.0
        _add_grad(u, [(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(I(2*i+2+t), 1.0), (I(2*i+t), -1.0)],
                 c, w, grad)

    # 6) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        w = A[94 + (i-1)]; c = SQRT2
        _add_grad(u, [(None, 1.0), (I(2*i+21+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(None, 1.0), (I(2*i+22+t), 1.0), (I(2*i+t), -1.0)],
                 c, w, grad)

    # 7) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        w = A[121 + (i-1)]; c = SQRT2
        _add_grad(u, [(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i+19+t), -1.0)],
                 [(None, 1.0), (I(2*i+20+t), 1.0), (I(2*i+2+t), -1.0)],
                 c, w, grad)

    return grad

def model4a_hessian(u, A, g, eps=1e-12) -> np.ndarray:
    """
    Compute the Hessian matrix of the model4a objective function at point u.
    Args:
        u (np.ndarray): The point at which to evaluate the Hessian.
        A (np.ndarray): The weights for the terms in the objective.
        g (np.ndarray): The linear coefficients in the objective.
        eps (float): Small value to avoid division by zero.
    Returns:
        np.ndarray: The Hessian matrix at point u.
    """
    u = np.asarray(u, float)
    A = np.asarray(A, float)
    H = np.zeros((u.size, u.size), dtype=float)

    # block 1
    # 1) i=1..10
    
    #pass (None,CONSTANT) to indicate constant term
    #pass (INDEX,COEFFICIENT) to indicate variable term, use sign of coefficient for +/- in term
    #pass terms sequewntially to build a and b, first a then b in expressiion sqrt(a^2 + b^2) using [] of tuples
    
    for i in range(1, 11):
        w = A[i-1]; c = 1.0
        _add_hess(u, [(I(2*i-1), 1.0)],
                 [(None, 1.0), (I(2*i), 1.0)],
                 c, w, H)

    # 2) i=1..9
    for i in range(1, 10):
        w = A[10 + (i-1)]; c = SQRT2
        _add_hess(u, [(None, 1.0), (I(2*i+1), 1.0)],
                 [(None, 1.0), (I(2*i+2), 1.0)],
                 c, w, H)

    # 3) i=1..9
    for i in range(1, 10):
        w = A[19 + (i-1)]; c = SQRT2
        _add_hess(u, [(None, 1.0), (I(2*i-1), -1.0)],
                 [(None, 1.0), (I(2*i),   1.0)],
                 c, w, H)

    # 4) i=1..30
    for i in range(1, 31):
        w = A[28 + (i-1)]; c = 1.0
        _add_hess(u, [(None, 1.0), (I(2*i+20), 1.0), (I(2*i), -1.0)],
                 [(I(2*i+19), 1.0), (I(2*i-1), -1.0)],
                 c, w, H)

    # 5) i=1..36
    for i in range(1, 37):
        t = 2 * ((i-1)//9)
        w = A[58 + (i-1)]; c = 1.0
        _add_hess(u, [(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(I(2*i+2+t), 1.0), (I(2*i+t), -1.0)],
                 c, w, H)

    # 6) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        w = A[94 + (i-1)]; c = SQRT2
        _add_hess(u, [(None, 1.0), (I(2*i+21+t), 1.0), (I(2*i-1+t), -1.0)],
                 [(None, 1.0), (I(2*i+22+t), 1.0), (I(2*i+t), -1.0)],
                 c, w, H)

    # 7) i=1..27
    for i in range(1, 28):
        t = 2 * ((i-1)//9)
        w = A[121 + (i-1)]; c = SQRT2
        _add_hess(u, [(None, 1.0), (I(2*i+1+t), 1.0), (I(2*i+19+t), -1.0)],
                 [(None, 1.0), (I(2*i+20+t), 1.0), (I(2*i+2+t), -1.0)],
                 c, w, H)

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


