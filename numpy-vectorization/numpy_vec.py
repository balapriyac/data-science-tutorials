import numpy as np

# Slow: Loop-based filtering
data = np.random.randn(1000000)
result = []
for x in data:
    if x > 0:
        result.append(x * 2)
    else:
        result.append(x)
result = np.array(result)

# Fast: Boolean indexing
data = np.random.randn(1000000)
result = data.copy()
result[data > 0] *= 2

# Slow: Explicit loops
matrix = np.random.rand(1000, 500)
row_means = np.mean(matrix, axis=1)
centered = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    centered[i] = matrix[i] - row_means[i]

# Fast: Broadcasting
matrix = np.random.rand(1000, 500)
row_means = np.mean(matrix, axis=1, keepdims=True)

# Slow: Manual looping
def complex_transform(x):
    if x < 0:
        return np.sqrt(abs(x)) * -1
    else:
        return x ** 2

data = np.random.randn(10000)
results = np.array([complex_transform(x) for x in data])

# Cleaner: np.vectorize()
def complex_transform(x):
    if x < 0:
        return np.sqrt(abs(x)) * -1
    else:
        return x ** 2

vec_transform = np.vectorize(complex_transform)
data = np.random.randn(10000)
results = vec_transform(data)

# Matrix multiplication the standard way
A = np.random.rand(100, 50)
B = np.random.rand(50, 80)
C = np.dot(A, B)

# Batch matrix multiply - gets messy
batch_A = np.random.rand(32, 10, 20)
batch_B = np.random.rand(32, 20, 15)
results = np.zeros((32, 10, 15))
for i in range(32):
    results[i] = np.dot(batch_A[i], batch_B[i])


centered = matrix - row_means

# Slow: Conditional logic in loops
temps = np.random.uniform(-10, 40, 100000)
classifications = []
for t in temps:
    if t < 0:
        classifications.append('freezing')
    elif t < 20:
        classifications.append('cool')
    else:
        classifications.append('warm')

# Fast: np.where() and np.select()
temps = np.random.uniform(-10, 40, 100000)
classifications = np.select(
    [temps < 0, temps < 20, temps >= 20],
    ['freezing', 'cool', 'warm'],
    default='unknown' # Added a string default value
)

# For simple splits, np.where() is cleaner:
scores = np.random.randint(0, 100, 10000)
results = np.where(scores >= 60, 'pass', 'fail')

# Slow: Loop-based gathering
lookup_table = np.array([10, 20, 30, 40, 50])
indices = np.random.randint(0, 5, 100000)
results = []
for idx in indices:
    results.append(lookup_table[idx])
results = np.array(results)

lookup_table = np.array([10, 20, 30, 40, 50])
indices = np.random.randint(0, 5, 100000)
results = lookup_table[indices]

matrix = np.arange(20).reshape(4, 5)
row_indices = np.array([0, 2, 3])
col_indices = np.array([1, 3, 4])
values = matrix[row_indices, col_indices]  # Gets matrix[0,1], matrix[2,3], matrix[3,4]

# Clean: einsum
A = np.random.rand(100, 50)
B = np.random.rand(50, 80)
C = np.einsum('ij,jk->ik', A, B)

# Batch matrix multiply - single line
batch_A = np.random.rand(32, 10, 20)
batch_B = np.random.rand(32, 20, 15)
results = np.einsum('bij,bjk->bik', batch_A, batch_B)

# Trace (sum of diagonal)
matrix = np.random.rand(100, 100)
trace = np.einsum('ii->', matrix)

# Transpose
transposed = np.einsum('ij->ji', matrix)

# Element-wise multiply then sum
A = np.random.rand(50, 50)
B = np.random.rand(50, 50)
result = np.einsum('ij,ij->', A, B)  # Same as np.sum(A * B)

# Slow: Manual row iteration
data = np.random.rand(1000, 50)
row_stats = []
for i in range(data.shape[0]):
    row = data[i]
    # Custom statistic not in NumPy
    stat = (np.max(row) - np.min(row)) / np.median(row)
    row_stats.append(stat)
row_stats = np.array(row_stats)

# Cleaner: apply_along_axis
data = np.random.rand(1000, 50)

def custom_stat(row):
    return (np.max(row) - np.min(row)) / np.median(row)

row_stats = np.apply_along_axis(custom_stat, axis=1, arr=data)

# Apply to each column
col_stats = np.apply_along_axis(custom_stat, axis=0, arr=data)
