import numpy as np

a = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
B = np.array([[7.0, 3.0], [2.0, 11.0], [9.0, 1.0]])
c = a[1:3] + B[:, 1]
print(c)