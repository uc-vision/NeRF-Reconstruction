import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, s):
    return 1/(1 + np.exp(-(s*x)))


def sigmoid_dir(x, s):
    return s*np.exp(-(s*x))/(1+np.exp(-(s*x)))**2


def neus_weight(t, sdf, s):
    alpha = np.maximum((sigmoid(sdf[:-1], s) - sigmoid(sdf[1:], s)) / sigmoid(sdf[:-1], s), 0)
    # alpha = np.concatenate((alpha, np.zeros(1,)))

    transmittance = np.cumprod(1 - alpha)
    transmittance = np.concatenate((np.zeros(1,), transmittance))

    alpha = np.concatenate((alpha, np.zeros(1,)))

    weight = transmittance * alpha

    density = sigmoid_dir(sdf, s) / sigmoid(sdf, s)

    print(np.sum(weight))
    
    return weight, density


if __name__ == "__main__":
    res = 100
    t = np.linspace(0, 10, res)
    sdf = np.concatenate([np.linspace(2, -2, int(res*0.4)), np.linspace(-2, 1, int(res*0.3)), np.linspace(1, -1, int(res*0.2)), np.linspace(-1, 0, int(res*0.1))])

    weight, density = neus_weight(t, sdf, 1)

    plt.plot(t, np.zeros(res))
    plt.plot(t, sdf)
    plt.plot(t, weight)
    plt.plot(t, density)
    plt.show()
    