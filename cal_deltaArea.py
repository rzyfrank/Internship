# cython:language_level=3

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


deltas = 2000
a = 215
r = 290
def fx(n):
    global deltas, a, r
    x1 = a+n
    x2 = a-n
    return deltas - (area(x2, r) - area(x1, r))

def fx1(n):
    a = 215
    r = 290

    x1 = a+n
    x2 = a-n
    return (area(x2, r) - area(x1, r))

def area(x,r):
    S = (r**2)*np.arccos(x/r) - x*np.sqrt(r**2 - x**2)
    return S

# x = [x * 2.77 for x in range(11)]
# x1 = [x * 0.02 for x in range(0, 11)]
# y = []
# for i in range(11):
#     a = x[i]
#     y.append(fx1(a)+4909)
#
# print(y)
#
# plt.plot(x1,y)
# plt.xlabel('offset_pixel')
# plt.ylabel('pixel_different')
# x_ticks = np.arange(0,0.23,0.02)
# plt.xticks(x_ticks)
# plt.show()
# plt.savefig('plot_fig.jpg')

# print(fx1(2.56))
# print(area(0, 290))

# print(fx(0))

def main(x1, x2, x3):
    global deltas, a, r
    deltas = x1
    a = x2
    r = x3


root = optimize.bisect(fx, 0, 30)
print(root)
root = root * 0.0024
# #
print(root)

