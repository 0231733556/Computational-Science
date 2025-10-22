import numpy as np
import logging as log

def set_logging_level(level):
    log.basicConfig(level=level)

# Helper to convert 1-based MATLAB k -> 0-based index
def I(k): return k-1

def _lin_values(u: np.ndarray, coeffs: list[tuple[int|None, float]]):
    """
    Build a dense coefficient vector c (same length as u) and compute s = c·u + const.
    coeffs: list of (index|None, coeff). Use (None, const) for constants.
    Returns: (c_vec, s)
    """
    u = np.asarray(u, float)
    c = np.zeros_like(u)
    c0 = 0.0
    for idx, coeff in coeffs:
        if idx is None:
            c0 += coeff
        else:
            c[idx] += coeff
    s = float(np.dot(c, u) + c0)
    return c, s

def _add_obj_pow(u: np.ndarray, coeffs, p: int, w: float, m: np.ndarray) -> None:
    """
    Add w * (c·u + c0)^p to scalar accumulator m (0-D ndarray).
    """
    _, s = _lin_values(u, coeffs)
    np.add(m, w * (s**p), out=m)

def _add_grad_pow(u: np.ndarray, coeffs, p: int, w: float, grad: np.ndarray) -> None:
    """
    Add gradient of w * (c·u + c0)^p to grad in-place:
        ∇ = w * p * (s)^(p-1) * c
    """
    c, s = _lin_values(u, coeffs)
    if p == 0:
        return
    np.add(grad, (w * p * (s**(p-1))) * c, out=grad)

def _add_obj_lin(u: np.ndarray, coeffs, w: float, m: np.ndarray) -> None:
    """
    Add linear term w * (c·u + c0) to m.
    """
    _, s = _lin_values(u, coeffs)
    np.add(m, w * s, out=m)

def _add_grad_lin(u: np.ndarray, coeffs, w: float, grad: np.ndarray) -> None:
    """
    Gradient of linear term w * (c·u + c0) is w * c.
    """
    c, _ = _lin_values(u, coeffs)
    np.add(grad, w * c, out=grad)

def obj(u: np.ndarray) -> float:
    """
    m(u) = -u_n + u_1^4 + sum_{j=2..n} (u_j - u_{j-1})^4
    Implemented via linear-form helpers, independent of n.
    """
    u = np.asarray(u, float)
    n = u.size
    if n < 1:
        raise ValueError("u must have length >= 1")
    m = np.array(0.0, dtype=float)

    # -u_n  (linear, w = -1, c = e_n)
    _add_obj_lin(u, [(I(n), 1.0)], w=-1.0, m=m)

    # u_1^4 (quartic, p=4, c = e_1)
    _add_obj_pow(u, [(I(1), 1.0)], p=4, w=1.0, m=m)

    # sum_{j=2..n} (u_j - u_{j-1})^4  (quartic, p=4, c = e_j - e_{j-1})
    for j in range(2, n+1):  # 0-based: j=1..n-1
        _add_obj_pow(u, [(I(j), 1.0), (I(j-1), -1.0)], p=4, w=1.0, m=m)

    return float(m)

def grad(u: np.ndarray) -> np.ndarray:
    """
    ∇m(u) computed modularly via linear forms.
    """
    u = np.asarray(u, float)
    n = u.size
    if n < 1:
        raise ValueError("u must have length >= 1")
    g = np.zeros_like(u)

    # -u_n  (linear, w = -1, c = e_n)
    _add_grad_lin(u, [(I(n), 1.0)], w=-1.0, grad=g)

    # u_1^4 (quartic, p=4, c = e_1)
    _add_grad_pow(u, [(I(1), 1.0)], p=4, w=1.0, grad=g)

    # sum_{j=2..n} (u_j - u_{j-1})^4  (quartic, p=4, c = e_j - e_{j-1})
    for j in range(2, n+1):  # 0-based: j=1..n-1
        _add_grad_pow(u, [(I(j), 1.0), (I(j-1), -1.0)], p=4, w=1.0, grad=g)

    return g

def _add_hess_pow(u: np.ndarray,
                  coeffs: list[tuple[int|None, float]],
                  p: int,
                  w: float,
                  H: np.ndarray) -> None:
    """
    Add Hessian of the term w * (c·u + c0)^p to H in-place:
        H += w * p * (p-1) * s^(p-2) * (c c^T)
    """
    if p < 2 or w == 0.0:
        return
    c, s = _lin_values(u, coeffs)
    scale = w * p * (p - 1) * (s ** (p - 2))
    # Outer product c c^T scaled
    np.add(H, scale * np.outer(c, c), out=H)

def hess(u: np.ndarray) -> np.ndarray:
    """
    Hessian of m(u) = -u_n + u_1^4 + sum_{j=2..n} (u_j - u_{j-1})^4.
    """
    u = np.asarray(u, float)
    n = u.size
    if n < 1:
        raise ValueError("u must have length >= 1")
    H = np.zeros((n, n), dtype=float)

    # -u_n → linear (p=1) → no Hessian

    # u_1^4  → p=4, c = e1
    _add_hess_pow(u, [(0, 1.0)], p=4, w=1.0, H=H)

    # (u_j - u_{j-1})^4 for j=2..n  → p=4, c = e_j - e_{j-1}
    for j in range(1, n):  # 0-based: j=1..n-1
        _add_hess_pow(u, [(j, 1.0), (j-1, -1.0)], p=4, w=1.0, H=H)

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


