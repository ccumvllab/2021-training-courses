
import matplotlib.pyplot as plt


def plot_polynomial(file, function, p_x, x=None, formula=""):
    pass


def plot_regression(file, x, y, prediction, groundtruth):
    plt.scatter(x , y ,color = '#7CFC00', s=15 , label='Data points')
    plt.plot(x, prediction[0][-1] * x + prediction[1][-1] , label='Prediction', color = 'b' , linewidth = 0.5 )
    plt.plot(x, groundtruth[0] * x + groundtruth[1] , label='Groundtruth', color = 'r' , linewidth = 0.5 )
    plt.legend(loc = 'upper left')
    plt.savefig(file)
    plt.show()





