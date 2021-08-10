import matplotlib.pyplot as plt
import numpy as np


def plot_polynomial(file, function, x, X=None):
    if X is None:
        raise TypeError('Expected `X = np.linspace(...)` must be passed.')

    Y_f = function(X)

    y = function(x)
    slope = y.grad
    Y_tangent = slope * (X - x) + y.value

    fig, ax = plt.subplots()
    ax.plot(X, Y_f.value)
    ax.plot(X, Y_tangent)
    fig.savefig(file)
