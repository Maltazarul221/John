import numpy as np

# A = [5, 9, 2]
# var = A[0]
#
# A2 = [[5, 9, 2]  , [8, 2, 0]]
# var2 = A2[0, 2]

# 1. Create a 3-D array with ones on the diagonal and zeros elsewhere
tensor_A = np.zeros((3, 3, 3))                             # functia primeste un parametru; cub cu valorile 0 (mai putin pe diagonala principala)
                                                           # 3 dimensiuni (axe: i, j, l), nr axe si nr de valori pe axe
for i in range(3):
    tensor_A[i, i, i] = 1                                  # acelasi indice la toate (de sus stanga pana jos dreapta) - putem deduce
print("#1 3-D array with ones on the diagonal:\n", tensor_A, "\n")


# 2. Create an array of 3×2 numbers, filled with ones
tensor_B = np.ones((3, 2))
print("#2 Array 3x2 filled with ones:\n", tensor_B, "\n")

# 3. Create a matrix of shape 2×5, filled with 6s
tensor_C = np.full((2, 5), 6)
print("#3 Matrix 2x5 filled with 6s:\n", tensor_C, "\n")

# 4. Create an array of 5, 13, 21, 29, ..., 101
tensor_D = np.arange(5, 102, 8)
print("#4 Array 5, 13, 21, ..., 101:\n", tensor_D, "\n")

# 5. Create a 1-D array of 20 evenly spaced elements between 7. and 12., inclusive
tensor_E = np.linspace(7, 12, 20)
print("#5 1-D array of 20 evenly spaced elements 7 to 12:\n", tensor_E, "\n")

# 6. Create a 2-D array (4×4) whose diagonal equals [1, 2, 3, 4] and with 0's elsewhere
tensor_F = np.diag([1, 2, 3, 4])
print("#6 4x4 array with diagonal [1,2,3,4]:\n", tensor_F, "\n")

# 7. Let x be an ndarray with dimensions [10, 10, 3] and all elements equal one.
# Reshape x so that it creates a 2-D array with the second dimension = 150
tensor_G = np.ones((10, 10, 3))                                             # 300 elements
tensor_G_reshaped = tensor_G.reshape(-1, 150)
print("#7 Reshape x to 2D with second dimension 150:\n",tensor_G_reshaped, "\n")

# 8. Let x be array [[1, 2, 3], [4, 5, 6]]. Convert it to [1 4 2 5 3 6]
tensor_H = np.array([[1, 2, 3], [4, 5, 6]])
x = np.column_stack(tensor_H)
y = x.flatten()
print("#8 My tensor:\n", tensor_H, "\n")
print("#8 Interleaved order:\n", x, "\n")
print("#8 Flatten x in interleaved order:\n", y, "\n")

# 9. Let X = np.arange(1, 5).reshape((2, 2)) and calculate the determinant of X
tensor_I = np.arange(1, 5).reshape((2, 2))
det_tensor_I = np.linalg.det(tensor_I)
print("#9 Determinant of X:\n", det_tensor_I, "\n")           # Determinant diferit de 0 => sistemul de ecuatii are solutii + are si matrice inversa

# 10. Return the sum along the diagonal of A = np.linspace(2, 16, 16).reshape(4, 4) (doar pentru matrici patratice)
name = np.linspace(2, 16, 16).reshape(4, 4)
sum_diag = np.trace(name)
print("#10 Sum along diagonal of A:\n", sum_diag, "\n")

# 11. Find the inverse of np.array([[1., 2.], [3., 4.]])
tensor_J = np.array([[1., 2.], [3., 4.]])
inv_tensor_J = np.linalg.inv(tensor_J)                        # Inversa anuleaza matricea initiala (daca * matricea cu matricea inversa obtin matricea unitate:
                                                              # 1 pe diag p si 0 in rest)
print("#11 Inverse of [[1.,2.],[3.,4.]]:\n",inv_tensor_J, "\n")

# 12. For vectors x2 = [4,5,6,13] and y2 = [5,0,6,11], check which elements of x are greater than y
x2 = np.array([4,5,6,13])
y2 = np.array([5,0,6,11])
greater_than_y2 = x2 > y2
print("#12 Elements of x greater than y:\n", greater_than_y2, "\n")  # Displays the mask
