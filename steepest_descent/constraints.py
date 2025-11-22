from math import sqrt
from functions import I
import numpy as np
import logging as log


def model4a_constraint_8_16(dx, dy, R):
    val = sqrt(dx*dx + dy*dy)- R
    return val

def model4a_constraint_8_17(dx, dy, R):
 
    val = sqrt(dx*dx + dy*dy) - R
    return val if val > 0.0 else 0.0
    
def model4a_constraint_8_18(dx, dy, R):

    val = sqrt(dx*dx + dy*dy)-R
    return val if val < 0.0 else 0.0

def make_penalized_model4a(fun, grad, constraint, xR, yR, R, k, j_range=range(31, 41)):
    """
    Create a penalized objective and its gradient for enforcing circular proximity constraints.

    This function builds and returns two callables (pen_fun, pen_grad) that augment a
    base objective `fun` and its gradient `grad` with quadratic penalty terms. Each
    penalty corresponds to a constraint of the form

        c_j(u) = (Ax_j - u[ix])**2 + (By_j - u[iy])**2 - R

        Ax_j = xR - j + 31
        By_j = yR - 4
    and `ix = I(2*j-1)`, `iy = I(2*j)` are integer indices into the decision vector `u`.
    Note: the module-level index-mapping function I and numpy must be available in the
    enclosing scope for the returned functions to work.

    The penalized objective is

        Phi(u) = fun(u) + k * sum_j c_j(u)**2,

    and its gradient is the gradient of `fun` plus the penalty contributions. For each
    constraint the penalty gradient contribution is

        d/du[ix] ( k * c_j(u)**2 ) = -4 * k * c_j(u) * dx_j
        d/du[iy] ( k * c_j(u)**2 ) = -4 * k * c_j(u) * dy_j

    with dx_j = Ax_j - u[ix] and dy_j = By_j - u[iy].

    
    Parameters
    ----------
    fun : callable
        Base objective function. Must accept a 1-D array-like `u` and return a scalar.
    grad : callable
        Function that returns the gradient of `fun` at `u`. Must accept the same `u`
        and return a sequence or array of shape compatible with the indices produced by `I`.
    xR, yR : float
        Reference coordinates used to form the target points (Ax_j, By_j) for each constraint.
    R : float
        Constant subtracted from squared distance in each constraint. (Interpreted as in
        the code: c_j = dx^2 + dy^2 - R.)
    k : float
        Penalty multiplier. Nonnegative scalar controlling the weight of constraint violations.
    j_range : iterable of int, optional
        Iterable producing the integer indices `j` used to construct constraints. The default
        is range(31, 41). The iterable will be converted to a list and iterated in order.

    - The returned functions close over `fun`, `grad`, `xR`, `yR`, `R`, `k`, `j_range`
      and require the module-level index mapping function `I` and numpy (`np`) to be
      available at call time.
    - Inputs are not modified in-place: both pen_fun and pen_grad convert `u` with
      numpy.asarray(...) and pen_grad works on a copy of the base gradient.
    - The code assumes that indices produced by `I` are valid for the arrays returned
      by `grad(u)` and that `u` is indexable at those positions.
    - If `fun`, `grad`, or `I` are missing or return incompatible shapes/types, the
      returned functions may raise NameError, TypeError, IndexError or ValueError.
      
    Returns
    -------
    pen_fun : callable
        Penalized objective function. Accepts a 1-D array-like `u` and returns a scalar.
    pen_grad : callable
        Gradient of the penalized objective. Accepts the same `u` and returns a 1-D array.

    Example
    Assuming `I` is defined and numpy imported as `np`:

        pen_fun, pen_grad = make_penalized_model4a(my_fun, my_grad, xR=10.0, yR=5.0, R=4.0, k=1e3)
        val = pen_fun(u0)
        g   = pen_grad(u0)

    """
    j_range = list(j_range)

    def penalty_terms(u):
        """
        Compute penalty terms for a set of constraints based on the decision vector `u`.

        For each index `j` in the external iterable `j_range`, this function:
        - obtains two indices into `u` via the external index-mapping function `I`:
            ix = I(2*j - 1) and iy = I(2*j)
        - constructs a target point (Ax, By) where Ax = xR - j + 31 and By = yR - 4,
            using the external scalar values `xR` and `yR`
        - computes the displacement from the point stored in `u` to the target:
            dx = Ax - u[ix], dy = By - u[iy]
        - evaluates the quadratic constraint value c = dx*dx + dy*dy - R,
            using the external scalar `R`
        - collects the tuple (c, dx, dy, ix, iy) for each j

        Parameters
        ----------
        u : array-like
                Decision vector (or any sequence) containing the variables. It will be
                converted to a numpy.ndarray of dtype float internally. The vector must be
                indexable at positions produced by the external function `I`.

        Returns
        -------
        list of tuples
                A list of tuples, one per `j` in `j_range`. Each tuple has the form
                (c, dx, dy, ix, iy) where
                - c (float): the penalty/constraint value equal to squared distance minus R
                - dx (float): x-component of the displacement (Ax - u[ix])
                - dy (float): y-component of the displacement (By - u[iy])
                - ix (int-like): index into `u` for the x-coordinate
                - iy (int-like): index into `u` for the y-coordinate

        Notes
        -----
        - The function depends on the external names `j_range`, `I`, `xR`, `yR`, and `R`
            being defined in the enclosing scope.
        - Complexity is linear in the number of elements in `j_range`.
        - No mutations are performed on `u`; only a read-only view is used after
            conversion to a numpy array.
        """
        u = np.asarray(u, float)
        c_vals = []
        for j in j_range:
            ix = I(2*j-1)    
            iy = I(2*j)     
            Ax = xR - j + 31
            By = yR - 4
            dx = Ax - u[ix]
            dy = By - u[iy]
            c=constraint(dx, dy, R)
            c_vals.append((c, dx, dy, ix, iy))
        return c_vals

    def pen_fun(u):
        """
        Compute a penalized objective value for a candidate variable vector.

        This function converts the input `u` to a NumPy array of floats, evaluates the base
        objective `fun(u)`, and adds penalty contributions computed from `penalty_terms(u)`.
        Each penalty term is expected to be a tuple of the form (c, dx, dy, ix, iy); only
        the first element `c` is used and contributes k * c**2 to the returned value.
        The variables `fun`, `penalty_terms`, and `k` are taken from the surrounding scope.

        Parameters
        ----------
        u : array_like
            Candidate variable vector. Will be converted to numpy.ndarray of dtype float
            via ``np.asarray(u, float)`` before evaluation.

        Returns
        -------
        float
            Scalar value equal to float(fun(u)) plus the sum of k * c**2 for every penalty
            term (c, dx, dy, ix, iy) produced by ``penalty_terms(u)``.

        Notes
        -----
        - The function coerces the base objective value to float before accumulating penalties.
        - Only the first element `c` of each penalty tuple is used; other tuple components
          (dx, dy, ix, iy) are ignored in the computation.
        - `fun`, `penalty_terms`, and `k` must be defined in the enclosing scope; if they are
          missing or return values of incompatible types, a NameError, TypeError, or ValueError
          may be raised.

        Examples
        --------
        Assuming `fun`, `penalty_terms`, and `k` are defined in the outer scope:
        >>> pen_val = pen_fun([1.0, 2.0, 3.0])
        """
        u = np.asarray(u, float)
        base = float(fun(u))
        val = base
        for c, _, _, _, _ in penalty_terms(u):
            val += 0.5 * k * c**2
        return val

    def pen_grad(u):
        """
        Compute the gradient of the objective including penalty contributions.

        This function computes the gradient of the underlying objective at the
        point `u` by calling the module-level `grad(u)` and then adding the
        contributions from penalty terms returned by `penalty_terms(u)`. Each
        penalty term is expected to be an iterable (c, dx, dy, ix, iy) where
        `c` is the scalar constraint value, `dx` and `dy` are partial
        derivatives of that constraint with respect to two degrees of freedom,
        and `ix`, `iy` are integer indices into the flattened gradient array
        where those derivatives apply.

        The contribution from a single penalty term is applied to the gradient
        entries at `ix` and `iy` as:

        Notes:
        - `u` is converted to a NumPy array of dtype float before use.
        - The base gradient is obtained by calling the module-level `grad(u)`.
        - Penalty terms are obtained by calling the module-level `penalty_terms(u)`.
        - The scalar `k` used in the penalty contribution is taken from the
            surrounding module scope.
        - The returned gradient is a new NumPy array (a copy), leaving the
            original `u` untouched.

        Parameters
        ----------
        u : array_like
                Input point at which the gradient is evaluated. Can be any array-like
                object convertible to a 1-D NumPy float array.

        Returns
        -------
        numpy.ndarray
                1-D array of floats containing the gradient of the objective with
                penalty contributions applied. The length and indexing correspond to
                those used by `grad` and `penalty_terms`.

        Examples
        --------
        >>> # assuming grad, penalty_terms and k are defined appropriately in the module
        >>> u = [0.1, 0.2, 0.3]
        >>> g = pen_grad(u)  # returns numpy array of same length as grad(u)
        """
        u = np.asarray(u, float)
        g = np.asarray(grad(u), float).copy()
        for c, dx, dy, ix, iy in penalty_terms(u):
            # If constraint value is zero (e.g. clamped constraints) no contribution
            if c == 0.0:
                continue
            # r = sqrt(dx^2 + dy^2)
            r = np.hypot(dx, dy)
            # avoid division by zero; when r == 0, dx and dy are also zero so contribution is zero
            if r == 0.0:
                continue
            # Gradient of 0.5 * k * c^2 is k * c * (dc/du).
            # c = sqrt(dx^2 + dy^2) - R, dc/d(u_ix) = -dx / r, dc/d(u_iy) = -dy / r
            factor = k * c / r
            g[ix] += - factor * dx
            g[iy] += - factor * dy
        return g

    return pen_fun, pen_grad


def make_lagrangian_model4a(fun, grad, hess, constraint, xR, yR, R, j_range=range(31, 41)):
    """
    Create Lagrangian helpers for model4a circular proximity constraints.

    Returns a tuple of four callables:
      - L_fun(u, lam): scalar Lagrangian value = fun(u) + sum_j lam_j * c_j(u)
      - L_grad_u(u, lam): gradient of L w.r.t. u (1-D array)
      - L_grad_lambda(u, lam): vector of constraint values c_j(u)
      - update_lambda(lam, u, rho=1.0, project_nonneg=True): simple dual ascent update

    Notes
    -----
    - c_j(u) is the raw constraint defined as sqrt(dx**2 + dy**2) - R where
      dx = Ax_j - u[ix], dy = By_j - u[iy] and Ax_j = xR - j + 31, By_j = yR - 4.
    - The returned functions require `I` and `np` to be available in module scope.
    - `lam` should be an array-like of length equal to number of j in `j_range`.
    """
    j_range = list(j_range)

    def _constraint_raw(u):
        u = np.asarray(u, float)
        c_list = []
        coords = []
        for j in j_range:
            ix = I(2*j-1)
            iy = I(2*j)
            Ax = xR - j + 31
            By = yR - 4
            dx = Ax - u[ix]
            dy = By - u[iy]
            c = constraint(dx, dy, R)
            c_list.append(c)
            coords.append((dx, dy, ix, iy))
        return np.asarray(c_list, float), coords

    def L_fun(u, lam):
        """Lagrangian value L(u, lam) = fun(u) + lam^T c(u)"""
        u = np.asarray(u, float)
        lam = np.asarray(lam, float)
        base = float(fun(u))
        c_vec, _ = _constraint_raw(u)
        return base + float(np.dot(lam, c_vec))

    def L_grad_u(u, lam):
        """Gradient of L with respect to u: grad(fun) + sum_j lam_j * dc_j/du"""
        u = np.asarray(u, float)
        lam = np.asarray(lam, float)
        g = np.asarray(grad(u), float).copy()
        _, coords = _constraint_raw(u)
        for idx, (dx, dy, ix, iy) in enumerate(coords):
            lam_j = lam[idx]
            # derivative of c = sqrt(dx^2+dy^2)-R wrt u[ix], u[iy]
            r = np.hypot(dx, dy)
            if r == 0.0:
                # when r == 0 the directional derivative is zero (dx=dy=0)
                continue
            dc_du_ix = - dx / r
            dc_du_iy = - dy / r
            g[ix] += lam_j * dc_du_ix
            g[iy] += lam_j * dc_du_iy
        return g

    def L_grad_lambda(u, lam=None):
        """Return the vector of constraint values c_j(u).

        This is the gradient of L w.r.t. the multipliers (dual variables).
        """
        u = np.asarray(u, float)
        c_vec, _ = _constraint_raw(u)
        return c_vec

    def update_lambda(lam, u, rho=1.0, project_nonneg=True):
        """Simple dual-ascent multiplier update: lam <- lam + rho * c(u).

        If `project_nonneg` is True, the updated multipliers are projected to
        the non-negative orthant (useful for inequality constraints of the
        form c(u) <= 0).
        """
        lam = np.asarray(lam, float).copy()
        c_vec = L_grad_lambda(u)
        lam += rho * c_vec
        if project_nonneg:
            lam = np.maximum(lam, 0.0)
        return lam

    def L_hess(u, lam):
        """Hessian of L w.r.t. u: H_fun(u) + sum_j lam_j * Hessian(c_j)(u)

        `hess` is the base Hessian callable passed to the factory and must
        return a full (n,n) array compatible with `u`.
        """
        u = np.asarray(u, float)
        lam = np.asarray(lam, float)
        H = np.asarray(hess(u), float).copy()
        _, coords = _constraint_raw(u)
        for idx, (dx, dy, ix, iy) in enumerate(coords):
            lam_j = lam[idx]
            if lam_j == 0.0:
                continue
            r = np.hypot(dx, dy)
            if r == 0.0:
                continue
            s = np.array([dx, dy], float)
            # local 2x2 Hessian for c = sqrt(dx^2+dy^2) - R
            H_local = (np.eye(2) / r) - np.outer(s, s) / (r**3)
            # add lam_j * H_local into global Hessian at (ix,iy)
            H[np.ix_([ix, iy], [ix, iy])] += lam_j * H_local
        return H

    return L_fun, L_grad_u, L_grad_lambda, update_lambda, L_hess


def lagrangian_solver(fun, grad, hess, constraint,algorithm, xR, yR, R,
                      u0, lam0=None, tol=1e-12, rho=1.0, max_iter=20,
                      j_range=range(31, 41)):
    """
    Solve the equality-constrained problem using an alternating primal-dual
    Lagrangian scheme that performs Newton primal steps.

    Parameters
    - fun, grad, hess : callables
        Base objective and its derivatives (hess returns full (n,n) array).
    - constraint : callable
        Constraint function with signature constraint(dx, dy, R) as used in
        `make_lagrangian_model4a`.
    - xR, yR, R : floats
        Constraint parameters used to assemble the constraint terms.
    - u0 : array_like
        Initial primal iterate.
    - lam0 : array_like or None
        Initial multipliers; if None zeros are used.
    - tol : float
        Tolerance on constraint norm for termination.
    - rho : float
        Dual step-length multiplier.
    - max_iter : int
        Maximum number of primal-dual iterations.
    - j_range : iterable
        Indices used to construct constraints.

    Returns
    - u, lam, c_vec, iters
        Final primal vector, multipliers, last constraint vector, and iterations used.
    """
    L_fun, L_grad_u, L_grad_lambda, update_lambda, L_hess = make_lagrangian_model4a(
        fun, grad, hess, constraint, xR, yR, R, j_range=j_range
    )

    u = np.asarray(u0, float).copy()
    if lam0 is None:
        lam = np.zeros(len(list(j_range)))
    else:
        lam = np.asarray(lam0, float).copy()

    for it in range(1, max_iter+1):
        # primal: Newton minimization of L(u, lam)
        fL = lambda uu: L_fun(uu, lam)
        gL = lambda uu: L_grad_u(uu, lam)
        hL = lambda uu: L_hess(uu, lam)
        res = algorithm(fL, gL, hL, u, tol)
        u = res[0]

        # dual update (equality constraints): lam <- lam + rho * c(u)
        c_vec = L_grad_lambda(u)
        lam = lam + rho * c_vec

        if np.linalg.norm(c_vec) < tol:
            return u, lam, c_vec, it
        
    log.info(f"Lagrangian solver finished, found u: {u} and lambda: {lam}")
    return u, lam, c_vec, max_iter


