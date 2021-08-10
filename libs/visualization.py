import matplotlib.pyplot as plt

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
