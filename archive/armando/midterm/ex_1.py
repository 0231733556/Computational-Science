import numpy as np
from algorithms import steepest_descent_conjugate, set_logging_level
import functions as fn
import logging as log


def __main__():
    set_logging_level(log.DEBUG)
    
    alpha_init = 1
    n_reset = 20
    r = 0.5
    c = 0.5
    f = lambda x: fn.obj(x)
    g = lambda x: fn.grad(x)
    
    
    n = 3
    u = np.arange(1, n+1)
    steepest_descent_conjugate(f, g, u, alpha_init=alpha_init, n=n_reset, r = r, c = c, beta=fn.beta_1)
    
    n = 30
    u = np.arange(1, n+1)
    steepest_descent_conjugate(f, g, u, alpha_init=alpha_init, n=n_reset, r = r, c = c, beta=fn.beta_1)
    
    n = 300
    u = np.arange(1, n+1)
    steepest_descent_conjugate(f, g, u, alpha_init=alpha_init, n=n_reset, r = r, c = c, beta=fn.beta_1)

    
if __name__ == "__main__":
    __main__()