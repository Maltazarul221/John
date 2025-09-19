import numpy as np

# 11. Creează două matrice A și B de dimensiuni 3x3
# Aici definim matricele explicit, fiecare cu 3 rânduri și 3 coloane
A = np.array([[3, 4, 5], [1, 2, 3], [4, 5, 6]])
B = np.array([[-2, 5, 1], [7, 0, 2], [-1, 0, 5]])
print("#11 Matrice A și B:\n", "A:\n", A, "\nB:\n", B, "\n")

# 12. Suma tuturor elementelor din A
# np.sum(A) returnează suma tuturor elementelor din matrice
sum_A = np.sum(A)
print("#12 Suma elementelor A:", sum_A, "\n")

# 13. Suma elementelor pe coloane (axis=0)
# axis=0 → operăm vertical, obținem suma fiecărei coloane
sum_cols = np.sum(A, axis=0)
print("#13 Suma pe coloane A:", sum_cols, "\n")

# 14. Suma elementelor pe rânduri (axis=1)
# axis=1 → operăm orizontal, obținem suma fiecărui rând
sum_rows = np.sum(A, axis=1)
print("#14 Suma pe rânduri A:", sum_rows, "\n")

# 15. Operații element-wise între A și B
# Adunare, scădere, multiplicare element cu element (Hadamard product)
add_AB = A + B
sub_AB = A - B
mul_AB = A * B
print("#15 A + B:\n", add_AB, "\n")
print("#15 A - B:\n", sub_AB, "\n")
print("#15 A * B:\n", mul_AB, "\n")

# 16. Multiplicarea scalară și element-wise cu dimensiuni diferite
# 2 * A → multiplicare scalară (fiecare element înmulțit cu 2)
scalar_mul = 2 * A
# A * B[0] → vector (prima linie din B) se broadcast pe toate rândurile lui A
elementwise_broadcast_row = A * B[0]
# A * B[:,1] → coloana a doua din B se broadcast pe toate coloanele lui A
# reshape(-1,1) transformă coloana într-o matrice compatibilă pentru broadcasting
elementwise_broadcast_col = A * B[:,1].reshape(-1,1)
print("#16 2 * A:\n", scalar_mul, "\n")
print("#16 A * B[0]:\n", elementwise_broadcast_row, "\n")
print("#16 A * B[:,1]:\n", elementwise_broadcast_col, "\n")

# 17. Produs matricial între A și B
# np.matmul(A,B) sau A @ B realizează produsul clasic de matrice
matmul_AB = np.matmul(A, B)
print("#17 Produs matricial A @ B:\n", matmul_AB, "\n")

# 18. Transpunerea matricei A
# A.T inversează rândurile cu coloanele
transpose_A = A.T
print("#18 Transpunere A:\n", transpose_A, "\n")

# 19. Reshape A într-un vector 1x9
# reshape(1,9) creează o matrice cu 1 rând și 9 coloane
reshape_row = A.reshape(1,9)
print("#19 Reshape A 1x9:\n", reshape_row, "\n")

# 20. Reshape A într-un vector coloană
# reshape(-1,1) lasă numpy să calculeze automat numărul de rânduri
reshape_col = A.reshape(-1,1)
print("#20 Reshape A -1x1:\n", reshape_col, "\n")

# 21. Diagonală și trace
# np.diagonal() extrage diagonală principală
# np.trace() calculează suma elementelor diagonalei
diagonal_A = A.diagonal()
trace_A = A.trace()
print("#21 Diagonală A:\n", diagonal_A, "\n")
print("#21 Trace A:", trace_A, "\n")

# 22. Flatten A într-un vector 1D
# flatten() creează o copie a matricei sub formă de vector 1D
flatten_A = A.flatten()
print("#22 Flatten A:\n", flatten_A, "\n")

# 23. Statistici descriptive pentru matricea A
max_A = np.max(A)       # valoarea maximă
min_A = np.min(A)       # valoarea minimă
mean_A = np.mean(A)     # media aritmetică
var_A = np.var(A)       # varianța - Măsoară cât se împrăștie valorile în jurul mediei - în medie, elementele sunt ≈var_A unități departe de medie.
std_A = np.std(A)       # deviația standard - Reprezintă mărimea medie a abaterii valorilor față de medie
print("#23 Max A:", max_A)
print("#23 Min A:", min_A)
print("#23 Mean A:", mean_A)
print("#23 Var A:", var_A)
print("#23 Std A:", std_A, "\n")

# 24. Creează doi vectori și calculează norma euclidiană și dot product
vec1 = np.array([6, -8, 0])
vec2 = np.array([0, 0, -2])
# Norma euclidiană (L2 norm) calculează lungimea vectorului
norm_vec1 = np.linalg.norm(vec1)
norm_vec2 = np.linalg.norm(vec2)
# Produsul scalar (dot product) între doi vectori - un indicator de orientare și aliniere între vectori
# Pozitiv → vectorii tind să fie în aceeași direcție.
# Zero → vectorii sunt perpendiculare.
# Negativ → vectorii tind să fie în direcții opuse.
dot_product = np.dot(vec1, vec2)
print("#24 Norma euclidiană vec1:", norm_vec1)
print("#24 Norma euclidiană vec2:", norm_vec2)
print("#24 Dot product vec1 · vec2:", dot_product, "\n")
