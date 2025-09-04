# INSTALL SciPy
from scipy.spatial import distance

x = [3,6,9]
y = [1,0,1]

print(distance.euclidean(x,y))
# Output >> 10.198039027185569

print(distance.cityblock(x,y))
# Output >> 16

print(distance.minkowski(x,y,p=3))
# Output >> 9.028714870948003

print(distance.minkowski(x,y,p=1))
# Output >> 16.0

print(distance.minkowski(x,y,p=2))
# Output >> 10.198039027185569
