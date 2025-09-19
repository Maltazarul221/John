#Scalar
#A scalar is a single number. In NumPy, a scalar can be created using a zero-dimensional array.

import numpy as np

A = np.array(8)
print(A)

#Vector
#A vector is a one-dimensional array of numbers or a sequence. It can be represented as:
# Row Vector

A = np.array([3,6,8,1])
print(A)

#Colum Vector

A = np.array([[3] , [6], [8] , [1]])
print(A)

#Intrinsic NumPy functions
#There are many built-in functions allowing for a fast creation routine;
#Vector declaration using numerical ranges

# evenly spaced values within an interval
C = np.arange(2,8)
print(C)

## evenly spaced numbers over an interval
D = np.linspace(0,1,5)
print(D)