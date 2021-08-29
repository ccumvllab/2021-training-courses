import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_polynomial(file, function, p_x, x=None, formula=""):

    y = function(x).value
    p_y = function(p_x).value

    p_grad = function(p_x).grad
    tangent_y = x * p_grad + p_y - p_grad * p_x

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Q1: Polynomial and Gradient")

    plt.plot(
        x, y, label=f"$f(x)={formula}$")
    plt.plot(x, tangent_y, label=f"the tangent at the point $p={p_x}$")
    plt.plot(p_x, p_y, 'o', label=f"the point $p$")

    plt.legend(loc="best")
    plt.savefig(file, dpi=150)
    plt.show()


def plot_regression(file, x, y, prediction, groundtruth):

    # get polynomial
    x_line = np.linspace(np.min(x)-0.5, np.max(x)+0.5)
    y_p = prediction[0][-1]*x_line+prediction[1][-1]
    y_g = groundtruth[0]*x_line+groundtruth[1]

    # init plot parameters
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Q2: Linear Regression")

    # draw lines and points
    plt.plot(x, y, 'o', label="Data points", color="lime", markersize=4)
    plt.plot(
        x_line, y_g, label=f"Groundtruth : $y={groundtruth[0]}x+{groundtruth[1]}$", color="red")
    plt.plot(
        x_line, y_p, label=f"Prediction : $y={prediction[0][-1]:.3f}x+{prediction[1][-1]:.3f}$")

    plt.legend(loc="best")
    plt.savefig(file, dpi=150)
    plt.show()
