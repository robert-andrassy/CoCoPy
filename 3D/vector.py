import numpy as np

def divergence(f, *varargs):
    D = np.array(np.gradient(f, *varargs)[1:])
    return np.trace(D)

def curl(f, *varargs):
    _, dx, dy, dz = np.gradient(f, *varargs)
    return np.array([dy[2] - dz[1], dz[0] - dx[2], dx[1] - dy[0]])
