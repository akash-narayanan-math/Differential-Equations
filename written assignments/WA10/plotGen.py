import math
import matplotlib.pyplot as plt
import numpy as np


# Unit step function
def step(t0, x):
    return 0.5 * (np.sign(x - t0) + 1)


# Solution to differential equation
def y_sol(t):
    y_hat = 1 / 2 * math.e ** (-2 * t) + 1 / 2
    p = step(1, t) * (math.e ** (-(t - 1)) - math.e ** (-2*(t - 1)))
    return y_hat + p


def main():
    # set x-coordinates
    x = np.arange(0, 5, 0.01)
    plt.xlabel("t")

    # set corresponding y-coordinates
    y = y_sol(x)
    plt.ylabel("y")

    # plot points
    plt.plot(x, y)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.savefig('graph.png', transparent=True)
    plt.show()


if __name__=='__main__':
    main()
