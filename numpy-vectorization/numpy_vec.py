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

