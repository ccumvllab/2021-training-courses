import matplotlib.pyplot as plt
import numpy as np


def plot_polynomial(file, function, p_x, x=None):
    if x is None:
        raise TypeError('Expected `x = np.linspace(...)` must be passed.')

    y = function(x)
    p_y = function(p_x)
    slope = p_y.grad
    y_tangent = slope * (x - p_x) + p_y.value

    fig, ax = plt.subplots()
    ax.plot(x, y.value)
    ax.plot(x, y_tangent)
    ax.legend(['f(x)', 'the tangent at the point p'])
    fig.savefig(file)
