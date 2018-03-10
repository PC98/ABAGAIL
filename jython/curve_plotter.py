import matplotlib.pyplot as plt
import os, pickle

with open('learning_curve_data.pickle', 'rb') as f:
    data = pickle.load(f)
    x = range(38, 537, 38)
    for key in data:
        plt.grid()
        plt.plot(x, data[key][0], 'o-', color="r")
        plt.plot(x, data[key][1], 'o-', color="g")
        plt.xlabel('# of Training Examples')
        plt.ylabel('Accuracy')
        plt.title('%s ANN Learning Curve' % key)
        plt.legend(['Training', 'Testing'], loc="best")
        plt.savefig(os.path.join('.', 'data', 'plot', "ANN %s.png" % key))
        plt.close()

with open('num_iterations_data.pickle', 'rb') as f:
    data = pickle.load(f)
    x = range(100, 1600, 150)
    for key in data:
        plt.grid()
        plt.plot(x, data[key][0], 'o-', color="r")
        plt.plot(x, data[key][1], 'o-', color="g")
        plt.xlabel('# of Iterations')
        plt.ylabel('Accuracy')
        plt.title('%s ANN Num Iterations Curve' % key)
        plt.legend(['Training', 'Testing'], loc="best")
        plt.savefig(os.path.join('.', 'data', 'plot', "ANN Iterations %s.png" %
                                 key))
        plt.close()

with open('four_peaks_data.pickle', 'rb') as f:
    data = pickle.load(f)
    x = range(200, 3200, 200)
    plt.grid()
    plt.plot(x, data['RHC'], 'o-', color="r")
    plt.plot(x, data['SA'], 'o-', color="g")
    plt.plot(x, data['GA'], 'o-', color="b")
    plt.plot(x, data['MIMIC'], 'o-', color="c")
    plt.xlabel('# of Iterations')
    plt.ylabel('Optimal Value')
    plt.title('Four Peaks Num Iterations Curve')
    plt.legend(['RHC', 'SA', 'GA', "MIMIC"], loc="best")
    plt.savefig(os.path.join('.', 'data', 'plot', "Four Peaks Iterations.png"))
    plt.close()


with open('count_ones_data.pickle', 'rb') as f:
    data = pickle.load(f)
    x = range(50, 550, 50)
    plt.grid()
    plt.plot(x, data['RHC'], 'o-', color="r")
    plt.plot(x, data['SA'], 'o-', color="g")
    plt.plot(x, data['GA'], 'o-', color="b")
    plt.plot(x, data['MIMIC'], 'o-', color="c")
    plt.xlabel('# of Iterations')
    plt.ylabel('Optimal Value')
    plt.title('Count Ones Num Iterations Curve')
    plt.legend(['RHC', 'SA', 'GA', "MIMIC"], loc="best")
    plt.savefig(os.path.join('.', 'data', 'plot', "Count Ones Iterations.png"))
    plt.close()


