# Be wary of floating point errors, I haven't gone through to correct them but
# the tables print relatively accurate values

import math
import numpy as np
import pandas as pd
from euler_method import euler

# Derivative expressed as a function of x and y
def derivative(x, y):
    return 1 + (3 / (x**3)) - (2 * y / x)
#    return (x ** 3 * math.e ** (-2 * x)) - 2 * y   # Practice function

# Returns the y value of the Euler method approximation using initial value and
# step size
def improved_euler(x0, y0, x, h):
    pointList = [(x0, y0)]
    for i in np.arange(x0, x, h):
        (x_old, y_old) = pointList[-1]
        x_n = x_old + h
        k_1 = derivative(x_old, y_old)
        k_2 = derivative(x_n, y_old + h * k_1)
        y_n = y_old + (h / 2) * (k_1 + k_2)
        pointList.append((x_n, y_n))
    return pointList

# Returns the actual value of the solution to the differential equation
def exact(x):
    return (1 / (3 * x ** 2)) * (9 * np.log(x) + x ** 3 + 2)
#    return (math.e ** (-2 * x) / 4) * (x ** 4 + 4) # Practice function

def main():
    d = {}
    x_0 = 1.0
    y_0 = 1.0
    x_n = 2.0
    step_size = 0.1
    list0 = euler(x_0, y_0, x_n, 0.1)
    list1 = euler(x_0, y_0, x_n, 0.05)
    list2 = euler(x_0, y_0, x_n, 0.025)
    list3 = improved_euler(x_0, y_0, x_n, 0.1)
    list4 = improved_euler(x_0, y_0, x_n, 0.05)
    list5 = improved_euler(x_0, y_0, x_n, 0.025)
    for i in np.arange(x_0, x_n + step_size, step_size):
        y0 = [point[1] for point in list0 if math.isclose(point[0], i)][0]
        y1 = [point[1] for point in list1 if math.isclose(point[0], i)][0]
        y2 = [point[1] for point in list2 if math.isclose(point[0], i)][0]
        y3 = [point[1] for point in list3 if math.isclose(point[0], i)][0]
        y4 = [point[1] for point in list4 if math.isclose(point[0], i)][0]
        y5 = [point[1] for point in list5 if math.isclose(point[0], i)][0]
        y6 = exact(i)
        entry = [y0, y1, y2, y3, y4, y5, y6]
        d[i] = entry

    with pd.option_context('display.precision', 9):
        datFrame = pd.DataFrame.from_dict(d, orient='index', 
                                          columns=['h = 0.1', 'h = 0.05', 
                                                   'h = 0.025', 'h = 0.1',
                                                   'h = 0.05', 'h = 0.025',
                                                   'Exact'])
        print(datFrame.to_latex())

if __name__=="__main__":
    main()
