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

# assert sum(vector) == l1_norm

inf_norm = linalg.norm(vector, ord=np.inf)
print(f"{inf_norm = }")

# Output >> inf_norm = 6.0

neg_inf_norm = linalg.norm(vector, ord=-np.inf)
print(f"{neg_inf_norm = }")

# Output >> neg_inf_norm = 1.0

another_vector = np.array([1,2,0,5,0])
l0_norm = linalg.norm(another_vector,ord=0)
print(f"{l0_norm = }")

# Output >> l0_norm = 3.0

matrix = vector.reshape(2,3)
print(matrix)

frob_norm = linalg.norm(matrix,ord='fro')
print(f"{frob_norm = :.2f}")

# Output >> frob_norm = 9.54

frob_norm = linalg.norm(matrix)
print(f"{frob_norm = :.2f}")

# Output >> frob_norm = 9.54

nuc_norm = linalg.norm(matrix,ord='nuc')
print(f"{nuc_norm = :.2f}")

# Output >> nuc_norm = 10.28

matrix_1_norm = linalg.norm(matrix,ord=1,axis=0)
print(f"{matrix_1_norm = }")

# Output >> matrix_1_norm = array([5., 7., 9.])

matrix_1_norm = linalg.norm(matrix,ord=1,axis=1)
print(f"{matrix_1_norm = }")

# Output >> matrix_1_norm = array([ 6., 15.])
