import numpy as np


vector = np.arange(1,7)
print(vector)

from numpy import linalg

l2_norm = linalg.norm(vector)
print(f"{l2_norm = :.2f}")

# Output >> l2_norm = 9.54

l2_norm = linalg.norm(vector, ord=2)
print(f"{l2_norm = :.2f}")

# Output >> l2_norm = 9.54

l1_norm = linalg.norm(vector, ord=1)
print(f"{l1_norm = :.2f}")

# Output >> l1_norm = 21.00


