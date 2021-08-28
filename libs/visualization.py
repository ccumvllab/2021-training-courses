import matplotlib.pyplot as plt
import matplotlib


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

    plt.plot(
        x, y, label=f"$f(x)={formula}$")
    plt.plot(x, tangent_y, label=f"the tangent at the point $p={p_x}$")
    plt.plot(p_x, p_y, 'o', label=f"the point $p$")

    plt.legend(loc="best")
    plt.savefig(file, dpi=150)
    plt.show()


def plot_regression(file, x, y, prediction, groundtruth):
    pass
