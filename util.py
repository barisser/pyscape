import numpy as np

def random_array_range(x, y):
    a = np.random.rand(x, y)
    c = np.ones((x, y))
    b = np.multiply(a, 2)
    d = np.subtract(b, c)
    return d

def random_array_range3d(x, y, z):
    a = np.random.rand(x, y, z)
    c = np.ones((x, y, z))
    b = np.multiply(a, 2)
    d = np.subtract(b, c)
    return d
