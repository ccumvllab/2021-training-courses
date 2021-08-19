import matplotlib.pyplot as plt

def plot_polynomial(file, function, p_x, x=None, formula=""):
    pass


def plot_regression(file, x, y, prediction, groundtruth):
    plt.figure(figsize=(8,6),dpi=150)
    gt_poly = groundtruth[0]*x + groundtruth[1]
    pd_poly = prediction[0][-1]*x + prediction[1][-1]
    plt.scatter(x, y, s=15, c='lime', label='Data points', alpha=0.7, zorder=2)
    plt.plot(x, pd_poly, c="blue", linewidth=1, label='Groundtruth', alpha=0.7, zorder=1)
    plt.plot(x, gt_poly, c="red", linewidth=1, label='Prediction', alpha=0.7, zorder=3)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.savefig(file)
    plt.show()