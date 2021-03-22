# Be wary of floating point errors, I haven't gone through to correct them but
# the tables print relatively accurate values

import math
import numpy as np
import pandas as pd

# Derivative expressed as a function of x and y
def derivative(x, y):
    return 1 + (3 / (x**3)) - (2 * y / x)

# Returns the y value of the Euler method approximation using initial value and
# step size
def euler(x0, y0, x, h):
    pointList = [(x0, y0)]
    for i in np.arange(x0, x, h):
        (x_old, y_old) = pointList[-1]
        x_n = x_old + h
        y_n = y_old + h * derivative(x_old, y_old)
        pointList.append((x_n, y_n))
    return pointList

# Returns the actual value of the solution to the differential equation
def exact(x):
    return (1 / (3 * x ** 2)) * (9 * np.log(x) + x ** 3 + 2)

def main():
    d = {}
    x_0 = 1.0
    y_0 = 1.0
    x_n = 2.0
    step_size = 0.1
    list0 = euler(x_0, y_0, x_n, 0.1)
    list1 = euler(x_0, y_0, x_n, 0.05)
    list2 = euler(x_0, y_0, x_n, 0.025)
    for i in np.arange(x_0, x_n + step_size, step_size):
        y0 = [point[1] for point in list0 if math.isclose(point[0], i)][0]
        y1 = [point[1] for point in list1 if math.isclose(point[0], i)][0]
        y2 = [point[1] for point in list2 if math.isclose(point[0], i)][0]
        y3 = exact(i)
        entry = [y0, y1, y2, y3]
        d[i] = entry

    print("{:<8} {:<15} {:<15} {:<15} {:<15}".format('x', 'h = 0.1', 'h = 0.05',
           'h = 0.025', 'Exact'))
    for x, entry in d.items():
        y0, y1, y2, y3 = entry
        print("{:<8.1f} {:<15.9f} {:<15.9f} {:<15.9f} {:<15.9f}".format(x, y0,
            y1, y2, y3))
    with pd.option_context('display.precision', 9):
        datFrame = pd.DataFrame.from_dict(d, orient='index', 
                                          columns=['h = 0.1', 'h = 0.05', 
                                                   'h = 0.025', 'Exact'])
        print(datFrame.to_latex())

if __name__=="__main__":
    main()
