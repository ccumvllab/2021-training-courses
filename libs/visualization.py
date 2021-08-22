import matplotlib.pyplot as plt


def plot_polynomial(file, function, p_x, x=None, formula=""):
    pass


def plot_regression(file, x, y, prediction, groundtruth):
    plt.scatter(x, y, linewidths=0.1, c="green", label='Data points')
    m_gt, b_gt = groundtruth[0], groundtruth[1]
    gt = x * m_gt + b_gt
    m, b = prediction[0], prediction[1]
    y_hat = x * m[-1] + b[-1]
    plt.plot(x, gt, c="red", label='Groundtruth')
    plt.plot(x, y_hat, c="blue", label='Prediction')
    plt.legend(loc='upper left')
    plt.savefig(file)
