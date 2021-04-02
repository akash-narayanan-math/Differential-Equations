import math
import pandas as pd
import matplotlib.pyplot as plt

# Constants for the problem
b = 6
w = 2


# Returns the value of x'(t) at a given position
def x_derivative(x, y):
    return (-b * x) / (math.sqrt(x ** 2 + y ** 2))


# Returns the value of y'(t) at a given position
def y_derivative(x, y):
    return (-b * y) / (math.sqrt(x ** 2 + y ** 2)) + w


# Returns the value of y'(x) at a given position
def dy_dx(x, y):
    dy = y_derivative(x, y)
    dx = x_derivative(x, y)
    if not math.isclose(dx, 0):  # Prevents division by zero
        return dy / dx           # error at x-nullcline
    else:
        return 0


# Returns the position of the drone after one step
# given an initial position and step size
def next_pos(x, y, h):
    k_0 = dy_dx(x, y)
    k_1 = dy_dx(x + h / 2, y + k_0 / 2)
    k_2 = dy_dx(x + h / 2, y + k_1 / 2)
    k_3 = dy_dx(x + h, y + h * k_2)

    x_next = x + h
    y_next = y + h / 6 * (k_0 + 2 * k_1 + 2 * k_2 + k_3)
    return [x_next, y_next]


# Evaluates a list of points using the Runge-Kutta method
def runge_kutta(x_0, y_0, x, n):
    h = (x - x_0) / n  # Step size calculation
    point_list = [[x_0, y_0]]
    for i in range(n - 1):  # Limiting floating point errors
        init_pos = point_list[-1]
        next_point = next_pos(init_pos[0], init_pos[1], h)
        point_list.append(next_point)
    return point_list


def main():
    point_list = runge_kutta(5, 0, 0, 100)
    df = pd.DataFrame(point_list, columns=['x', 'y'])
    print(df)
    with pd.option_context('display.precision', 5):
        print(df.to_latex())

    df.plot(x='x', y='y', legend=None)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid()

    plt.savefig('rungeKutta2.png', transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
