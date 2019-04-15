import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# x = np.arange(-6, 6, 0.3)
# y = np.arange(-6, 6, 0.3)
#
# xgrid, ygrid = np.meshgrid(x, y)
# zgrid = xgrid ** 2 - ygrid ** 2
#
# print(xgrid, ygrid, zgrid, len(xgrid), len(ygrid), len(zgrid), sep='\n\n')

# frame = plt.figure(figsize=(10, 8))
#
# axes = Axes3D(frame)
#
# axes.plot_surface(
#     xgrid, ygrid, zgrid, cmap=plt.cm.Pastel2_r, linewidth=0.5, rstride=1, cstride=1)
#
# axes.set_xlabel('$X$')
# axes.set_ylabel('$Y$')
# axes.set_zlabel('$Z$')
# axes.set_title('$F(X,Y)=X^2 − Y^2$')
#
# plt.show()

def test1(X, y):
    Z = np.array(X)

    x = np.arange(len(X))
    y = np.arange(len(X[0]))
    x, y = np.meshgrid(x, y)

    frame = plt.figure(figsize=(10, 8))
    axes = Axes3D(frame)
    axes.plot_surface(x, y, Z.T)
    plt.show()

def test2(X, y):
    for i in range(len(X)):
        pds = pd.Series(X[i])
        pds.plot()
    plt.show()
