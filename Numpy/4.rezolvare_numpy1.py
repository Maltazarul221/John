import numpy as np

# 1. Scalar: un singur număr, 0D (fără dimensiuni)
scalar = np.array(10)  # np.array(x) creează un array NumPy; aici x=10 → scalar
print("#1 Scalar:", scalar, "\n")

# 2. Vector rând și coloană: 1D vs 2D
row_vector = np.array([4, 7, 1, 9])        # vector 1D (rând)
column_vector = np.array([[4], [7], [1], [9]])  # vector 2D (coloană)
print("#2 Row vector:", row_vector)
print("#2 Column vector:\n", column_vector, "\n")

# 3. np.arange(start, stop[, step]): generează valori consecutive
# start=5, stop=16 (stop nu este inclus), pas implicit 1
vector_range = np.arange(5, 16)
print("#3 Vector arange 5-15:", vector_range, "\n")

# 4. np.linspace(start, stop, num): generează num valori echidistante între start și stop
# start=0, stop=2, num=6 → 6 valori de la 0 la 2 inclusiv
vector_linspace = np.linspace(0, 2, 6)
print("#4 Vector linspace 0-2:", vector_linspace, "\n")

# 5. Indexare condițională
players_height = np.array([1.72, 1.65, 1.88, 1.91, 1.82, 1.76, 1.80])

# 5a. np.where(condiție) returnează un tuplu cu array-uri de indici
tall_indices = np.where(players_height > 1.8)
print("#5a Indici jucători >1.8m:", tall_indices, "\n")

# 5b. Indexarea directă: extrage valorile care respectă condiția
tall_heights = players_height[players_height > 1.8]
print("#5b Înălțimi jucători >1.8m:", tall_heights, "\n")

# 6. Matrice 2x3 (2 rânduri x 3 coloane)
matrix_2x3 = np.array([[5, 8, 2], [1, 6, 9]])
# 6a. Elementul de pe rândul 1, coloana 2 (indexare: [rând, coloană])
print("#6a Element rând 1, col 2:", matrix_2x3[1, 2])
# 6b. Rând complet sau coloană completă (slice notation)
print("#6b Rând 0:", matrix_2x3[0])
print("#6b Coloana 2:", matrix_2x3[:, 2], "\n")

# 7. Matrice 4x6 generată din np.arange() și reshaped
# np.arange(1,25) → valori 1..24, reshape(4,6) → matrice 4 rânduri x 6 coloane
matrix_4x6 = np.arange(1, 25).reshape(4, 6)
print("#7 Matrice 4x6:\n", matrix_4x6, "\n")

# 8. Matrice 3x2 din np.linspace() și reshape
# np.linspace(0,1,6) → 6 valori echidistante între 0 și 1, reshape(3,2) → matrice 3 rânduri x 2 coloane
matrix_3x2 = np.linspace(0, 1, 6).reshape(3, 2)
print("#8 Matrice 3x2 linspace:\n", matrix_3x2, "\n")

# 9. Tensor 3D: 2x3x2
# np.arange(1,13) → valori 1..12, reshape(2,3,2) → tensor cu 3 dimensiuni
tensor_3d = np.arange(1, 13).reshape(2, 3, 2)
print("#9 Tensor 3D 2x3x2:\n", tensor_3d, "\n")

# 10. Extrage primul „plan” complet din tensor (primele 3x2 valori)
# tensor_3d este un tensor 3D cu forma (2, 3, 2):
#   2 planuri (layers)
#   3 rânduri pe plan
#   2 coloane pe plan
# Sintaxa tensor_3d[0, :, :] înseamnă:
#   0   → primul plan (layer 0, numărat de la 0)
#   :   → toate rândurile din planul 0
#   :   → toate coloanele din planul 0
# Rezultatul este o matrice 2D cu toate rândurile și coloanele primului plan
sub_tensor = tensor_3d[0, :, :]
print("#10 Sub-tensor primul plan:\n", sub_tensor, "\n")
