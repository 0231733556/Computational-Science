from math import sqrt
from functions import I

def model4a_constraint_8_16(u,j,R,y_R,x_R):
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
    assert j in range(31,41), "j must be in 31..40"
    
    part_1 = (x_R- j +31 -u[I(2*j-1)])**2
    part_2 = (y_R - 4 - u[I(2*j)] )**2
    return sqrt(part_1+part_2) == R

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
