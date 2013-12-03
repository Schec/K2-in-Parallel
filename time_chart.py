import time
import jodys_serial_v2 as k2code
import matplotlib.pyplot as plt
import numpy as np


def make_feature_chart():
    ftimes = []
    features = range(1,102,10)

    for i in range(len(features)):
        myarray = np.random.binomial(1,0.9, size=(100,features[i]))
        node_order = list(range(features[i]))
        start = time.time()
        k2code.k2(myarray,node_order)
        end = time.time()
        ftimes.append(end-start)

    plt.plot(features,ftimes)
    plt.xlabel('Number of features')
    plt.ylabel('Serial Time')
    plt.show()

def make_observation_chart():
    otimes = []
    observations = range(1,10002,1000)

    for i in range(len(observations)):
        myarray = np.random.binomial(1,0.9, size=(observations[i],30))
        node_order = list(range(30))
        start = time.time()
        k2code.k2(myarray,node_order)
        end = time.time()
        otimes.append(end-start)

    plt.plot(observations,otimes)
    plt.xlabel('Number of observations')
    plt.ylabel('Serial Time')
    plt.show()

if __name__ == '__main__':
    make_feature_chart()
    make_observation_chart()