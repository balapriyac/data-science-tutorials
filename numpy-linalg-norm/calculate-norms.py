import numpy as np


vector = np.arange(1,7)
print(vector)

from numpy import linalg

l2_norm = linalg.norm(vector)
print(f"{l2_norm = :.2f}")
