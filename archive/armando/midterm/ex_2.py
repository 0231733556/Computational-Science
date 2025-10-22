import numpy as np
from algorithms import newtons_method, set_logging_level
import functions as fn
import logging as log



def __main__():
    set_logging_level(log.DEBUG)
    
    tol = 1e-11
    f = lambda x: fn.obj(x)
    g = lambda x: fn.grad(x)
    hess = lambda x: fn.hess(x)
    
    n = 3
    u = np.arange(1, n+1)
    newtons_method(f, g, hess, u, tol=tol)
    # f(u) : -1.4174111811317323; u: [0.62996052 1.25992105 1.88988157]
    
    n = 30
    u = np.arange(1, n+1)
    newtons_method(f, g, hess, u, tol=tol)
    # fun(u) : -14.174111811317355
    
    n = 300
    u = np.arange(1, n+1)
    newtons_method(f, g, hess, u, tol=tol)
    # fun(u) : -141.7411181131724
    
if __name__ == "__main__":
    __main__()