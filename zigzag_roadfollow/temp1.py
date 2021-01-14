import numpy as np

l = np.array([[255, 255, 0, 255],
             [255, 255, 0, 0]])

r = l.nonzero()
print(r)

# array([0, 0, 0, 1, 1]), array([0, 1, 3, 0, 1])