import matplotlib.pyplot as plt

def plot_polynomial(file, function, p_x, x=None, formula=""):
    pass


def plot_regression(file, x, y, prediction, groundtruth):
    plt.scatter(x,y,s = 8,color = 'green',label = 'Data points')
    plt.plot(x,prediction[0][-1]*x+prediction[1][-1],linewidth = 0.5,color = 'blue',label = 'Prediction')
    plt.plot(x,groundtruth[0]*x+groundtruth[1],linewidth = 0.5,color = 'red',label = 'Groundtruth')
    plt.legend(loc = 'upper left')
    plt.savefig(file)
    plt.show()
