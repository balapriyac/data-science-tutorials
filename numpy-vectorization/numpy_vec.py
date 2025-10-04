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
