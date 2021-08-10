import matplotlib.pyplot as plt
import numpy as np


def plot_polynomial(file, function, x):
    X = np.linspace(0, 100)
    Y_f = function(X)

    y = function(x)
    slope = y.grad
    Y_tangent = slope * (X - x) + y.value

    fig, ax = plt.subplots()
    ax.plot(X, Y_f.value)
    ax.plot(X, Y_tangent)
    fig.savefig(file)
