import matplotlib.pylab as plt
import numpy as np
x = np.array([1,2,3,4,5,6])
y = np.array([1,3,4,5,6,7])
m = np.array([str(i+1) for i in range(30)])
print(m)
unique_markers = set(m)  # or yo can use: np.unique(m)

for um in unique_markers:
    mask = m == um
    # mask is now an array of booleans that van be used for indexing
    plt.scatter(x[mask], y[mask], marker=um)
plt.show()
