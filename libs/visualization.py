def plot_polynomial(file, function, p_x, x=None, formula=""):
    pass


def plot_regression(file, x, y, prediction, groundtruth):
    # Data 散佈圖
    plt.scatter(x, y, s=10, alpha=0.5, color='green', label='Data points')
    # Real Line
    m_gt, b_gt = groundtruth[0], groundtruth[1]
    plt.plot(x, m_gt*x + b_gt, alpha=0.5, color='red',
             label='Groundtruth', linewidth=1)
    # Predict Line
    m, b = prediction[0], prediction[1]
    plt.plot(x, m[-1] * x + b[-1], alpha=0.5, color='blue',
             label='Prediction', linestyle='--', linewidth=1)

    plt.legend(loc='best')  # 標籤放置
    plt.savefig(file, dpi=300)  # 在show之前要先save 解析度改為300
    plt.show()  # show圖
