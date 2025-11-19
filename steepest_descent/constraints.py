from math import sqrt
from functions import I
import numpy as np

def model4a_constraint_8_16(u, j, R, y_R, x_R, domain):
    """
    Check the model4a constraint 8.16: enforce that a computed Euclidean distance equals R.

    This function evaluates the constraint
        sqrt((x_R - j + 31 - u[I(2*j - 1)])**2 + (y_R - 4 - u[I(2*j)])**2) == R
    and returns a boolean indicating whether the equality holds.

    Parameters
    ----------
    u : Sequence[float] or ndarray
        Container of decision variables. Elements are accessed via indices produced
        by the callable I (see Notes).
    j : int
        Index parameter that selects which entries of u are used. Must be in the
        inclusive range 31..40. An AssertionError is raised otherwise.
    R : float
        Target radius (right-hand side of the distance equality).
    y_R : float
        Reference y-coordinate used in the distance computation.
    x_R : float
        Reference x-coordinate used in the distance computation.
    domain : float
        List of indices of j

    Returns
    -------
    bool
        True if the Euclidean distance defined above equals R, False otherwise.

    Raises
    ------
    AssertionError
        If j is not in range(31, 41).
    NameError, TypeError, IndexError
        If the symbol I is not defined as a callable in the surrounding scope,
        if I does not return a valid index for u, or if u is not indexable.

    Notes
    -----
    - The function depends on an external callable I that maps integer index
      expressions like 2*j-1 and 2*j to indices into u; I must be defined in the
      enclosing namespace where this function is used.
    - Using exact equality (==) on floating-point distances is numerically fragile.
      For numerical checks, prefer a tolerance-based comparison (e.g. math.isclose
      or abs(lhs - R) <= tol) instead of exact equality.
    """
    if j not in range(domain):
        return 0
    
    part_1 = (x_R - j + 31 -u[I(2*j-1)])**2
    part_2 = (y_R - 4 - u[I(2*j)] )**2
    c = sqrt(part_1+part_2) - R
    return c*c



# TODO: Change functions to return c^2
def model4a_constraint_8_17(u,j,R,y_R,x_R):
    """
    Check the model4a constraint 8.17: enforce that a computed Euclidean distance is less than or equal to R.

    This function evaluates the constraint
        sqrt((x_R - j + 31 - u[I(2*j - 1)])**2 + (y_R - 4 - u[I(2*j)])**2) <= R
    and returns a boolean indicating whether the inequality holds.
    Parameters
    ----------
    u : Sequence[float] or ndarray
        Container of decision variables. Elements are accessed via indices produced
        by the callable I (see Notes).
    j : int
        Index parameter that selects which entries of u are used. Must be in the
        inclusive range 31..40. An AssertionError is raised otherwise.
    R : float
        Target radius (right-hand side of the distance equality).
    y_R : float
        Reference y-coordinate used in the distance computation.
    x_R : float
        Reference x-coordinate used in the distance computation.

    Returns
    -------
    bool
        True if the Euclidean distance defined above equals R, False otherwise.

    Raises
    ------
    AssertionError
        If j is not in range(31, 41).
    NameError, TypeError, IndexError
        If the symbol I is not defined as a callable in the surrounding scope,
        if I does not return a valid index for u, or if u is not indexable.
    """
    assert j in range(31,41), "j must be in 31..40"
    part_1 = (x_R- j +31 -u[I(2*j-1)])**2
    part_2 = (y_R - 4 - u[I(2*j)] )**2
    return sqrt(part_1+part_2)<=R

def model4a_constraint_8_18(u,j,R,y_R,x_R):
    """
    Check the model4a constraint 8.18: enforce that a computed Euclidean distance is greater than or equal to R.

    This function evaluates the constraint
        sqrt((x_R - j + 31 - u[I(2*j - 1)])**2 + (y_R - 4 - u[I(2*j)])**2) >= R
    and returns a boolean indicating whether the inequality holds.
    Parameters
    ----------
    u : Sequence[float] or ndarray
        Container of decision variables. Elements are accessed via indices produced
        by the callable I (see Notes).
    j : int
        Index parameter that selects which entries of u are used. Must be in the
        inclusive range 31..40. An AssertionError is raised otherwise.
    R : float
        Target radius (right-hand side of the distance equality).
    y_R : float
        Reference y-coordinate used in the distance computation.
    x_R : float
        Reference x-coordinate used in the distance computation.

    Returns
    -------
    bool
        True if the Euclidean distance defined above equals R, False otherwise.

    Raises
    ------
    AssertionError
        If j is not in range(31, 41).
    NameError, TypeError, IndexError
        If the symbol I is not defined as a callable in the surrounding scope,
        if I does not return a valid index for u, or if u is not indexable.
    """
    assert j in range(31,41), "j must be in 31..40"
    part_1 = (x_R- j +31 -u[I(2*j-1)])**2
    part_2 = (y_R - 4 - u[I(2*j)] )**2
    return sqrt(part_1+part_2)>=R



def make_penalized_model4a(fun, grad, xR, yR, R, k, j_range=range(31, 41)):
    """
    Build penalized objective and gradient:
        Phi(u) = f(u) + k * sum_j c_j(u)^2
    where
        c_j(u) = (xR - j + 31 - u_{2j-1})^2 + (yR - 4 - u_{2j})^2 - R**2
    """
    j_range = list(j_range)

    def penalty_terms(u):
        u = np.asarray(u, float)
        c_vals = []
        for j in j_range:
            ix = I(2*j-1)    
            iy = I(2*j)     
            Ax = xR - j + 31
            By = yR - 4
            dx = Ax - u[ix]
            dy = By - u[iy]
            c = dx*dx + dy*dy - R
            c_vals.append((c, dx, dy, ix, iy))
        return c_vals

    def pen_fun(u):
        u = np.asarray(u, float)
        base = float(fun(u))
        val = base
        for c, dx, dy, ix, iy in penalty_terms(u):
            val += k * c**2
        return val

    def pen_grad(u):
        u = np.asarray(u, float)
        g = np.asarray(grad(u), float).copy()
        for c, dx, dy, ix, iy in penalty_terms(u):
            # gradient of k*c^2
            g[ix] += -4.0 * k * c * dx
            g[iy] += -4.0 * k * c * dy
        return g

    return pen_fun, pen_grad
