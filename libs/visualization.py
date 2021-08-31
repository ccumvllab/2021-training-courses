import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-bright')
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 150,
})


def plot_polynomial(file, function, p_x, x=None, formula=None):
    if x is None:
        raise TypeError('Expected `x = np.linspace(...)` must be passed.')

    if formula is None:
        raise TypeError('Expected `formula = "..."` must be passed.')

    y = function(x)
    p_y = function(p_x)
    slope = p_y.grad
    y_tangent = slope * (x - p_x) + p_y.value

    fig, ax = plt.subplots()
    ax.plot(x, y.value)
    ax.plot(x, y_tangent)
    ax.plot(p_x, p_y.value, 'o', markersize=3)
    ax.legend([rf'$f(x) = {formula}$', rf'the tangent at the point $p = {p_x}$', r'the point $p$'])
    fig.savefig(file)
    plt.close(fig)


def plot_regression(file, x, y, prediction, groundtruth):
    m, b = prediction
    m_gt, b_gt = groundtruth

    x_line = np.linspace(-10, 10)
    y_line = m[-1] * x_line + b[-1]
    y_line_gt = m_gt * x_line + b_gt

    fig, ax = plt.subplots()
    ax.plot(x_line, y_line, alpha=0.7)
    ax.plot(x, y, 'o', markersize=3, alpha=0.7)
    ax.plot(x_line, y_line_gt, alpha=0.7)
    ax.legend(['Prediction', 'Data points', 'Groundtruth'])
    fig.savefig(file)
    plt.close(fig)
