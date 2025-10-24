"""
================================================================================
                    GHID COMPLET NUMPY - Tutorial Detaliat
================================================================================

NumPy (Numerical Python) este biblioteca fundamentală pentru calcul științific
în Python. Oferă suport pentru arrays multidimensionale și o colecție vastă
de funcții matematice de nivel înalt.

Cuprins:
1. Instalare și Import
2. Arrays NumPy (ndarray)
3. Crearea Arrays-urilor
4. Proprietățile Arrays-urilor
5. Indexare și Slicing
6. Operații Matematice
7. Broadcasting
8. Funcții Universale (ufunc)
9. Manipularea Formei Arrays-urilor
10. Operații cu Axe
11. Sortare și Căutare
12. Operații Statistice
13. Algebra Liniară
14. Numere Aleatoare
15. Input/Output cu Fișiere
16. Performanță și Optimizare
================================================================================
"""

import numpy as np
import time

# ============================================================================
# 1. INSTALARE ȘI IMPORT
# ============================================================================

"""
Instalare:
    pip install numpy

Import standard:
    import numpy as np
"""

print("Versiunea NumPy:", np.__version__)
print("\n" + "="*80 + "\n")

# ============================================================================
# 2. ARRAYS NUMPY (ndarray)
# ============================================================================

"""
ndarray (N-dimensional array) este obiectul principal în NumPy.
Caracteristici:
- Toate elementele au același tip de date
- Dimensiune fixă la creare
- Operații vectorizate (rapide)
- Mai eficient decât listele Python pentru calcule numerice
"""

# Diferența între liste Python și arrays NumPy
lista_python = [1, 2, 3, 4, 5]
array_numpy = np.array([1, 2, 3, 4, 5])

print("Lista Python:", type(lista_python))
print("Array NumPy:", type(array_numpy))
print("Tipul elementelor:", array_numpy.dtype)
print("\n" + "="*80 + "\n")

# ============================================================================
# 3. CREAREA ARRAYS-URILOR
# ============================================================================

print("3. METODE DE CREARE A ARRAYS-URILOR\n")

# 3.1 Din liste sau tuple
arr1d = np.array([1, 2, 3, 4, 5])
print("Array 1D:", arr1d)

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Array 2D:\n", arr2d)

arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Array 3D:\n", arr3d)

# 3.2 Arrays cu valori inițiale
zeros = np.zeros((3, 4))  # Array cu zeros
print("\nZeros (3x4):\n", zeros)

ones = np.ones((2, 3, 4))  # Array cu unu
print("\nOnes (2x3x4) - forma:", ones.shape)

empty = np.empty((2, 2))  # Array neinitialized (rapid)
print("\nEmpty (2x2):\n", empty)

full = np.full((3, 3), 7)  # Array cu valoare specifică
print("\nFull cu 7 (3x3):\n", full)

# 3.3 Secvențe numerice
arange = np.arange(0, 10, 2)  # Similar cu range()
print("\nArange (0 la 10, pas 2):", arange)

linspace = np.linspace(0, 1, 5)  # 5 valori uniform distribuite
print("Linspace (0 la 1, 5 valori):", linspace)

logspace = np.logspace(0, 2, 5)  # Spațiere logaritmică
print("Logspace (10^0 la 10^2, 5 valori):", logspace)

# 3.4 Matrice speciale
identity = np.eye(4)  # Matrice identitate
print("\nMatrice identitate (4x4):\n", identity)

diag = np.diag([1, 2, 3, 4])  # Matrice diagonală
print("\nMatrice diagonală:\n", diag)

# 3.5 Specificarea tipului de date
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1, 2, 3], dtype=np.float64)
arr_complex = np.array([1+2j, 3+4j], dtype=np.complex128)

print("\nTipuri de date:")
print("int32:", arr_int.dtype)
print("float64:", arr_float.dtype)
print("complex128:", arr_complex.dtype)
print("\n" + "="*80 + "\n")

# ============================================================================
# 4. PROPRIETĂȚILE ARRAYS-URILOR
# ============================================================================

print("4. PROPRIETĂȚILE ARRAYS-URILOR\n")

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print("Array-ul nostru:\n", arr)
print("\nProprietăți:")
print("ndim (număr dimensiuni):", arr.ndim)
print("shape (forma):", arr.shape)
print("size (număr total elemente):", arr.size)
print("dtype (tip date):", arr.dtype)
print("itemsize (bytes per element):", arr.itemsize)
print("nbytes (total bytes):", arr.nbytes)
print("strides:", arr.strides)  # Pași în memorie
print("\n" + "="*80 + "\n")

# ============================================================================
# 5. INDEXARE ȘI SLICING
# ============================================================================

print("5. INDEXARE ȘI SLICING\n")

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 5.1 Indexare simplă
print("Element [0, 0]:", arr[0, 0])
print("Element [1, 2]:", arr[1, 2])
print("Ultimul element:", arr[-1, -1])

# 5.2 Slicing
print("\nPrima linie:", arr[0, :])
print("Prima coloană:", arr[:, 0])
print("Sub-array [0:2, 1:3]:\n", arr[0:2, 1:3])

# 5.3 Indexare booleană
mask = arr > 6
print("\nMască booleană (arr > 6):\n", mask)
print("Elemente > 6:", arr[mask])

# 5.4 Indexare fancy (cu liste/arrays)
indices = [0, 2]
print("\nLiniile 0 și 2:\n", arr[indices])
print("Elemente specifice:", arr[[0, 1, 2], [0, 1, 2]])  # Diagonala

# 5.5 Modificare prin indexare
arr_copy = arr.copy()
arr_copy[arr_copy > 6] = 0
print("\nArray modificat (>6 devine 0):\n", arr_copy)
print("\n" + "="*80 + "\n")

# ============================================================================
# 6. OPERAȚII MATEMATICE
# ============================================================================

print("6. OPERAȚII MATEMATICE\n")

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 6.1 Operații element-wise
print("a:", a)
print("b:", b)
print("\nAdunare a + b:", a + b)
print("Scădere a - b:", a - b)
print("Înmulțire a * b:", a * b)
print("Împărțire a / b:", a / b)
print("Putere a ** 2:", a ** 2)
print("Modulo a % 2:", a % 2)

# 6.2 Operații cu scalari (broadcasting)
print("\na + 10:", a + 10)
print("a * 2:", a * 2)
print("a / 2:", a / 2)

# 6.3 Funcții matematice universale
print("\nFuncții matematice:")
print("np.sqrt(a):", np.sqrt(a))
print("np.exp(a):", np.exp(a))
print("np.log(a):", np.log(a))
print("np.sin(a):", np.sin(a))
print("np.cos(a):", np.cos(a))

# 6.4 Operații de agregare
print("\nOperații de agregare:")
print("Sumă:", np.sum(a))
print("Medie:", np.mean(a))
print("Minim:", np.min(a))
print("Maxim:", np.max(a))
print("Std deviation:", np.std(a))
print("Varianță:", np.var(a))

# 6.5 Produse
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print("\nProdus element-wise:")
print(matrix1 * matrix2)

print("\nProdus matriceal (dot product):")
print(np.dot(matrix1, matrix2))
# sau
print(matrix1 @ matrix2)
print("\n" + "="*80 + "\n")

# ============================================================================
# 7. BROADCASTING
# ============================================================================

print("7. BROADCASTING\n")

"""
Broadcasting permite operații între arrays de forme diferite.
Regulile broadcasting:
1. Dacă arrays-urile au număr diferit de dimensiuni, forma celui mai mic
   este completată cu 1 în stânga
2. Arrays-urile sunt compatibile pe o dimensiune dacă au aceeași mărime
   sau una din ele are mărime 1
3. După broadcasting, fiecare array se comportă ca și cum ar avea forma
   maximă de-a lungul fiecărei dimensiuni
"""

# Exemplu 1: Scalar și array
arr = np.array([1, 2, 3])
print("arr:", arr)
print("arr + 5:", arr + 5)  # 5 este broadcast la [5, 5, 5]

# Exemplu 2: 1D și 2D
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

print("\nMatrix:\n", matrix)
print("Vector:", vector)
print("Matrix + Vector:\n", matrix + vector)

# Exemplu 3: Broadcasting cu reshape
arr1 = np.array([[1], [2], [3]])  # (3, 1)
arr2 = np.array([10, 20, 30])     # (3,)

print("\narr1 (3,1):\n", arr1)
print("arr2 (3,):", arr2)
print("arr1 + arr2:\n", arr1 + arr2)

# Exemplu 4: Operații complexe
a = np.arange(3).reshape(3, 1)  # (3, 1)
b = np.arange(3)                 # (3,)
print("\na:\n", a)
print("b:", b)
print("a + b (broadcasting):\n", a + b)
print("\n" + "="*80 + "\n")

# ============================================================================
# 8. FUNCȚII UNIVERSALE (ufunc)
# ============================================================================

print("8. FUNCȚII UNIVERSALE (ufunc)\n")

"""
ufunc = universal functions
Funcții vectorizate care operează element-wise pe arrays
Sunt mult mai rapide decât loop-urile Python
"""

x = np.array([1, 2, 3, 4, 5])

# 8.1 Funcții matematice
print("Funcții matematice:")
print("Original:", x)
print("np.abs (valoare absolută):", np.abs(x - 3))
print("np.sqrt:", np.sqrt(x))
print("np.square:", np.square(x))
print("np.exp:", np.exp(x))
print("np.log:", np.log(x))
print("np.log10:", np.log10(x))

# 8.2 Funcții trigonometrice
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print("\nFuncții trigonometrice:")
print("Unghiuri (radiani):", angles)
print("np.sin:", np.sin(angles))
print("np.cos:", np.cos(angles))
print("np.tan:", np.tan(angles))

# 8.3 Funcții de comparație
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

print("\nComparații:")
print("a:", a)
print("b:", b)
print("a > b:", a > b)
print("a == b:", a == b)
print("np.maximum(a, b):", np.maximum(a, b))
print("np.minimum(a, b):", np.minimum(a, b))

# 8.4 Funcții cu reducere
print("\nReduceri:")
print("np.add.reduce(a) (sumă):", np.add.reduce(a))
print("np.multiply.reduce(a) (produs):", np.multiply.reduce(a))

# 8.5 Funcții cu acumulare
print("\nAcumulări:")
print("np.add.accumulate(a):", np.add.accumulate(a))
print("np.multiply.accumulate(a):", np.multiply.accumulate(a))
print("\n" + "="*80 + "\n")

# ============================================================================
# 9. MANIPULAREA FORMEI ARRAYS-URILOR
# ============================================================================

print("9. MANIPULAREA FORMEI ARRAYS-URILOR\n")

arr = np.arange(12)
print("Array original:", arr)

# 9.1 Reshape
reshaped = arr.reshape(3, 4)
print("\nReshape (3, 4):\n", reshaped)

reshaped3d = arr.reshape(2, 2, 3)
print("\nReshape (2, 2, 3):\n", reshaped3d)

# 9.2 Flatten și ravel
print("\nFlatten (copie):", reshaped.flatten())
print("Ravel (view):", reshaped.ravel())

# 9.3 Transpose
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\nMatrix originală:\n", matrix)
print("Transpose:\n", matrix.T)
print("np.transpose:\n", np.transpose(matrix))

# 9.4 Adăugare dimensiuni
arr = np.array([1, 2, 3])
print("\nArray 1D:", arr, "shape:", arr.shape)
print("Cu np.newaxis:", arr[np.newaxis, :], "shape:", arr[np.newaxis, :].shape)
print("Cu np.expand_dims:", np.expand_dims(arr, axis=0).shape)

# 9.5 Squeeze (eliminare dimensiuni de mărime 1)
arr = np.array([[[1, 2, 3]]])
print("\nArray cu dimensiuni extra:", arr.shape)
print("După squeeze:", np.squeeze(arr).shape)

# 9.6 Concatenare
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("\nConcatenare pe axis=0 (vertical):\n", np.concatenate([a, b], axis=0))
print("\nConcatenare pe axis=1 (orizontal):\n", np.concatenate([a, b], axis=1))

# Alternative pentru concatenare
print("\nvstack:\n", np.vstack([a, b]))
print("\nhstack:\n", np.hstack([a, b]))

# 9.7 Split
arr = np.arange(16).reshape(4, 4)
print("\nArray original:\n", arr)

chunks = np.split(arr, 2)
print("\nSplit în 2 părți:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}")

# 9.8 Stack
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\nStack arrays:")
print("np.stack([a, b]):\n", np.stack([a, b]))
print("np.stack([a, b], axis=1):\n", np.stack([a, b], axis=1))
print("\n" + "="*80 + "\n")

# ============================================================================
# 10. OPERAȚII CU AXE
# ============================================================================

print("10. OPERAȚII CU AXE\n")

"""
În NumPy, axis specifică direcția de-a lungul căreia se execută operația:
- axis=0: de-a lungul liniilor (vertical)
- axis=1: de-a lungul coloanelor (orizontal)
- axis=2: de-a lungul adâncimii (pentru 3D)
"""

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print("Matrix:\n", matrix)

# 10.1 Sume pe axe
print("\nSumă totală:", np.sum(matrix))
print("Sumă pe axis=0 (coloane):", np.sum(matrix, axis=0))
print("Sumă pe axis=1 (linii):", np.sum(matrix, axis=1))

# 10.2 Medie pe axe
print("\nMedie pe axis=0:", np.mean(matrix, axis=0))
print("Medie pe axis=1:", np.mean(matrix, axis=1))

# 10.3 Min/Max pe axe
print("\nMaxim pe axis=0:", np.max(matrix, axis=0))
print("Minim pe axis=1:", np.min(matrix, axis=1))

# 10.4 Argmax/Argmin (indici)
print("\nIndice maxim pe axis=0:", np.argmax(matrix, axis=0))
print("Indice minim pe axis=1:", np.argmin(matrix, axis=1))

# 10.5 Operații cumulative
print("\nCumsum pe axis=0:\n", np.cumsum(matrix, axis=0))
print("\nCumsum pe axis=1:\n", np.cumsum(matrix, axis=1))

# 10.6 Keepdims (păstrează dimensiunile)
result = np.sum(matrix, axis=1, keepdims=True)
print("\nSumă cu keepdims:\n", result)
print("Shape:", result.shape)
print("\n" + "="*80 + "\n")

# ============================================================================
# 11. SORTARE ȘI CĂUTARE
# ============================================================================

print("11. SORTARE ȘI CĂUTARE\n")

# 11.1 Sortare
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("Array original:", arr)
print("Sortat:", np.sort(arr))
print("Sortat descrescător:", np.sort(arr)[::-1])

# Sort in-place
arr_copy = arr.copy()
arr_copy.sort()
print("Sort in-place:", arr_copy)

# 11.2 Argsort (indici de sortare)
indices = np.argsort(arr)
print("\nIndici de sortare:", indices)
print("Array sortat folosind indici:", arr[indices])

# 11.3 Sortare 2D
matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
print("\nMatrix originală:\n", matrix)
print("Sortare pe axis=0 (coloane):\n", np.sort(matrix, axis=0))
print("Sortare pe axis=1 (linii):\n", np.sort(matrix, axis=1))

# 11.4 Valori unice
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print("\nArray cu duplicate:", arr)
print("Valori unice:", np.unique(arr))
print("Valori unice cu counts:", np.unique(arr, return_counts=True))

# 11.5 Căutare
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("\nArray sortat:", arr)
print("np.where(arr > 5):", np.where(arr > 5))
print("Elemente > 5:", arr[np.where(arr > 5)])

# 11.6 Searchsorted (căutare binară)
print("\nSearchsorted (poziție inserare):")
print("Pentru 3.5:", np.searchsorted(arr, 3.5))
print("Pentru [2.5, 5.5, 8.5]:", np.searchsorted(arr, [2.5, 5.5, 8.5]))

# 11.7 Extract și nonzero
condition = arr % 2 == 0
print("\nElemente pare:")
print("np.extract:", np.extract(condition, arr))
print("Indici non-zero:", np.nonzero(condition))
print("\n" + "="*80 + "\n")

# ============================================================================
# 12. OPERAȚII STATISTICE
# ============================================================================

print("12. OPERAȚII STATISTICE\n")

data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

print("Date:\n", data)

# 12.1 Măsuri de tendință centrală
print("\nMăsuri de tendință centrală:")
print("Medie (mean):", np.mean(data))
print("Mediană (median):", np.median(data))

# 12.2 Măsuri de dispersie
print("\nMăsuri de dispersie:")
print("Varianță (variance):", np.var(data))
print("Deviație standard (std):", np.std(data))
print("Range:", np.ptp(data))  # peak to peak

# 12.3 Percentile și cuantile
print("\nPercentile:")
print("25th percentile:", np.percentile(data, 25))
print("50th percentile (mediană):", np.percentile(data, 50))
print("75th percentile:", np.percentile(data, 75))

quantiles = np.quantile(data, [0.25, 0.5, 0.75])
print("Cuartile:", quantiles)

# 12.4 Corelație și covarianță
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

print("\nx:", x)
print("y:", y)
print("\nMatrice de covarianță:\n", np.cov(x, y))
print("Coeficient de corelație:\n", np.corrcoef(x, y))

# 12.5 Histogramă
data_hist = np.random.randn(1000)
hist, bins = np.histogram(data_hist, bins=10)
print("\nHistogramă (10 bins):")
print("Frecvențe:", hist)
print("Margini bins:", bins)

# 12.6 Statistici pe axe
print("\nStatistici pe axis=0:")
print("Medie:", np.mean(data, axis=0))
print("Std:", np.std(data, axis=0))
print("\n" + "="*80 + "\n")

# ============================================================================
# 13. ALGEBRĂ LINIARĂ
# ============================================================================

print("13. ALGEBRĂ LINIARĂ\n")

# 13.1 Produs matriceal
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:\n", A)
print("\nMatrix B:\n", B)
print("\nProdus matriceal A @ B:\n", A @ B)
print("\nnp.matmul(A, B):\n", np.matmul(A, B))

# 13.2 Determinant
print("\nDeterminant A:", np.linalg.det(A))

# 13.3 Inversă
print("\nInversa A:\n", np.linalg.inv(A))
print("\nVerificare A @ inv(A):\n", A @ np.linalg.inv(A))

# 13.4 Rezolvare sisteme lineare (Ax = b)
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

print("\nSistem: Ax = b")
print("A:\n", A)
print("b:", b)
x = np.linalg.solve(A, b)
print("Soluție x:", x)
print("Verificare A @ x:", A @ x)

# 13.5 Valori și vectori proprii
A = np.array([[1, 2], [2, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nValori proprii:", eigenvalues)
print("Vectori proprii:\n", eigenvectors)

# 13.6 Norma
v = np.array([3, 4])
print("\nVector:", v)
print("Norma L2 (Euclidiană):", np.linalg.norm(v))
print("Norma L1:", np.linalg.norm(v, ord=1))
print("Norma infinită:", np.linalg.norm(v, ord=np.inf))

# 13.7 Rangul matricei
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nMatrix A:\n", A)
print("Rang:", np.linalg.matrix_rank(A))

# 13.8 Descompuneri
# SVD (Singular Value Decomposition)
A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, Vt = np.linalg.svd(A)
print("\nSVD decomposition:")
print("U shape:", U.shape)
print("s (valori singulare):", s)
print("Vt shape:", Vt.shape)

# QR decomposition
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
Q, R = np.linalg.qr(A)
print("\nQR decomposition:")
print("Q (ortogonală):\n", Q)
print("R (superior triunghiular):\n", R)
print("\n" + "="*80 + "\n")

# ============================================================================
# 14. NUMERE ALEATOARE
# ============================================================================

print("14. NUMERE ALEATOARE\n")

# 14.1 Seed pentru reproducibilitate
np.random.seed(42)
print("Cu seed=42:")
print("Random:", np.random.rand(3))

np.random.seed(42)  # Același seed
print("Cu seed=42 (din nou):", np.random.rand(3))

# 14.2 Generator modern (recomandat)
rng = np.random.default_rng(42)
print("\nGenerator modern:")
print("Random:", rng.random(3))

# 14.3 Distribuții uniforme
print("\nDistribuții uniforme:")
print("rand (0, 1):", np.random.rand(5))
print("uniform (-1, 1):", np.random.uniform(-1, 1, 5))
print("randint (0, 10):", np.random.randint(0, 10, 5))

# 14.4 Distribuții normale (Gaussiene)
print("\nDistribuții normale:")
print("randn (μ=0, σ=1):", np.random.randn(5))
print("normal (μ=10, σ=2):", np.random.normal(10, 2, 5))

# 14.5 Alte distribuții
print("\nAlte distribuții:")
print("exponential (λ=2):", np.random.exponential(2, 5))
print("poisson (λ=3):", np.random.poisson(3, 5))
print("binomial (n=10, p=0.5):", np.random.binomial(10, 0.5, 5))

# 14.6 Amestec aleator
arr = np.arange(10)
print("\nArray original:", arr)
np.random.shuffle(arr)
print("După shuffle:", arr)

# Permutare (returnează copie)
arr = np.arange(10)
print("Permutare:", np.random.permutation(arr))

# 14.7 Selecție aleatoare
arr = np.arange(100)
print("\nChoice (5 elemente):", np.random.choice(arr, 5))
print("Choice (5 elemente, fără replacement):",
      np.random.choice(arr, 5, replace=False))

# 14.8 Arrays multidimensionale aleatoare
print("\nMatrix 3x3 aleatoare:\n", np.random.rand(3, 3))
print("\nMatrix 3x3 normale:\n", np.random.randn(3, 3))
print("\n" + "="*80 + "\n")

# ============================================================================
# 15. INPUT/OUTPUT CU FIȘIERE
# ============================================================================

print("15. INPUT/OUTPUT CU FIȘIERE\n")

# 15.1 Salvare și încărcare arrays NumPy (.npy)
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Array de salvat:\n", arr)

# Salvare într-un fișier .npy
# np.save('array.npy', arr)
# Încărcare din fișier
# loaded_arr = np.load('array.npy')
print("(Salvare/Încărcare .npy - comentat pentru exemplu)")

# 15.2 Salvare și încărcare multiple arrays (.npz)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Salvare necomprimată
# np.savez('arrays.npz', a=arr1, b=arr2)
# Salvare comprimată
# np.savez_compressed('arrays_compressed.npz', a=arr1, b=arr2)
# Încărcare
# data = np.load('arrays.npz')
# print(data['a'], data['b'])
print("(Salvare/Încărcare .npz - comentat pentru exemplu)")

# 15.3 Salvare în format text
arr = np.array([[1, 2, 3], [4, 5, 6]])
# np.savetxt('array.txt', arr)
# np.savetxt('array.csv', arr, delimiter=',')
print("(Salvare .txt/.csv - comentat pentru exemplu)")

# Încărcare din text
# loaded = np.loadtxt('array.txt')
# loaded_csv = np.loadtxt('array.csv', delimiter=',')

# 15.4 Format CSV cu header
# np.savetxt('data.csv', arr, delimiter=',',
#            header='col1,col2,col3', comments='')

# 15.5 genfromtxt (pentru date mai complexe)
# data = np.genfromtxt('data.csv', delimiter=',',
#                      skip_header=1, filling_values=0)

print("\nNote: Operații I/O comentate pentru siguranță")
print("\n" + "="*80 + "\n")

# ============================================================================
# 16. PERFORMANȚĂ ȘI OPTIMIZARE
# ============================================================================

print("16. PERFORMANȚĂ ȘI OPTIMIZARE\n")

# 16.1 Comparație viteză: liste vs NumPy
size = 1000000

# Liste Python
python_list = list(range(size))
start = time.time()
result_list = [x * 2 for x in python_list]
time_list = time.time() - start

# NumPy array
numpy_array = np.arange(size)
start = time.time()
result_numpy = numpy_array * 2
time_numpy = time.time() - start

print(f"Liste Python: {time_list:.4f} secunde")
print(f"NumPy array: {time_numpy:.4f} secunde")
print(f"NumPy este {time_list/time_numpy:.1f}x mai rapid!")

# 16.2 Vectorizare vs loop-uri
def suma_loop(arr):
    """Sumă folosind loop Python"""
    total = 0
    for x in arr:
        total += x
    return total

arr = np.random.rand(100000)

start = time.time()
s1 = suma_loop(arr)
time_loop = time.time() - start

start = time.time()
s2 = np.sum(arr)
time_vectorized = time.time() - start

print(f"\nSumă cu loop: {time_loop:.4f} secunde")
print(f"Sumă vectorizată: {time_vectorized:.4f} secunde")
print(f"Vectorizarea este {time_loop/time_vectorized:.1f}x mai rapidă!")

# 16.3 Broadcasting vs operații explicite
matrix = np.random.rand(1000, 1000)
vector = np.random.rand(1000)

# Metoda ineficientă
start = time.time()
result = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    result[i] = matrix[i] + vector
time_loop = time.time() - start

# Metoda eficientă (broadcasting)
start = time.time()
result = matrix + vector
time_broadcast = time.time() - start

print(f"\nAdunare cu loop: {time_loop:.4f} secunde")
print(f"Adunare cu broadcasting: {time_broadcast:.4f} secunde")
print(f"Broadcasting este {time_loop/time_broadcast:.1f}x mai rapid!")

# 16.4 Copy vs View
arr = np.arange(1000000)

# View (rapid, nu copiază date)
start = time.time()
view = arr[::2]
time_view = time.time() - start

# Copy (mai lent, copiază date)
start = time.time()
copy = arr[::2].copy()
time_copy = time.time() - start

print(f"\nCreare view: {time_view:.6f} secunde")
print(f"Creare copy: {time_copy:.6f} secunde")

# 16.5 Verificare dacă este view sau copy
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
copy = arr[1:4].copy()

view[0] = 999
print(f"\nDupă modificare view:")
print(f"Array original: {arr}")
print(f"View: {view}")

arr_copy_test = np.array([1, 2, 3, 4, 5])
copy = arr_copy_test[1:4].copy()
copy[0] = 999
print(f"\nDupă modificare copy:")
print(f"Array original: {arr_copy_test}")
print(f"Copy: {copy}")

# 16.6 Sfaturi pentru optimizare
print("\n" + "-"*80)
print("SFATURI PENTRU OPTIMIZARE:")
print("-"*80)
print("""
1. Folosește operații vectorizate în loc de loop-uri Python
2. Profită de broadcasting pentru a evita tile/repeat
3. Folosește view-uri în loc de copii când este posibil
4. Alege tipul de date potrivit (float32 vs float64)
5. Evită conversii inutile între liste și arrays
6. Folosește funcții NumPy în-place când e posibil (arr += 1 vs arr = arr + 1)
7. Pentru operații complexe, consideră numba sau Cython
8. Folosește np.einsum pentru operații tensoriale complexe
9. Profită de funcțiile compiled (C) din NumPy
10. Măsoară întotdeauna performanța - nu presupune!
""")

# 16.7 Exemple de funcții in-place
arr = np.arange(10)
print("Array original:", arr)

# Out-of-place (creează array nou)
arr2 = arr + 1

# In-place (modifică array existent)
arr += 1
print("După operație in-place:", arr)

# 16.8 Einsum - operații tensoriale eficiente
# Produs scalar
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\nProdus scalar cu einsum: {np.einsum('i,i->', a, b)}")
print(f"Echivalent cu dot: {np.dot(a, b)}")

# Produs matriceal
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
C_einsum = np.einsum('ij,jk->ik', A, B)
C_matmul = A @ B
print(f"\nEinsum vs matmul sunt egale: {np.allclose(C_einsum, C_matmul)}")

# Transpunere
A = np.random.rand(3, 4, 5)
print(f"\nTranspunere cu einsum shape: {np.einsum('ijk->kji', A).shape}")

# 16.9 Alocări de memorie
print("\nConsiderații memorie:")
arr = np.zeros((1000, 1000))
print(f"Memorie ocupată: {arr.nbytes / 1024 / 1024:.2f} MB")
print(f"Tip date: {arr.dtype}")

# Economisire memorie cu float32
arr_float32 = np.zeros((1000, 1000), dtype=np.float32)
print(f"Memorie cu float32: {arr_float32.nbytes / 1024 / 1024:.2f} MB")
print(f"Economie: {(1 - arr_float32.nbytes/arr.nbytes)*100:.0f}%")

print("\n" + "="*80 + "\n")

# ============================================================================
# 17. TEHNICI AVANSATE
# ============================================================================

print("17. TEHNICI AVANSATE\n")

# 17.1 Fancy indexing avansat
arr = np.arange(12).reshape(3, 4)
print("Array:\n", arr)

# Selectare rânduri și coloane specifice
rows = np.array([0, 2])
cols = np.array([1, 3])
print("\nSelectare rânduri [0,2] și coloane [1,3]:")
print(arr[rows[:, np.newaxis], cols])

# 17.2 np.where cu trei argumente (if-then-else vectorizat)
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.where(arr > 3, arr * 2, arr / 2)
print(f"\nnp.where (dacă >3: *2, altfel: /2):")
print(f"Original: {arr}")
print(f"Rezultat: {result}")

# 17.3 np.select (multiple condiții)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
conditions = [arr < 3, (arr >= 3) & (arr < 7), arr >= 7]
choices = ['mic', 'mediu', 'mare']
result = np.select(conditions, choices)
print(f"\nnp.select (categorii):")
print(f"Original: {arr}")
print(f"Categorii: {result}")

# 17.4 np.meshgrid (grile 2D/3D)
x = np.linspace(-2, 2, 5)
y = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(x, y)

print(f"\nMeshgrid:")
print(f"x: {x}")
print(f"y: {y}")
print(f"X:\n{X}")
print(f"Y:\n{Y}")

# Calcul funcție 2D
Z = X**2 + Y**2
print(f"Z = X² + Y²:\n{Z}")

# 17.5 np.vectorize (vectorizare funcții Python)
def custom_function(x):
    if x < 0:
        return 0
    elif x < 5:
        return x * 2
    else:
        return x ** 2

vectorized_func = np.vectorize(custom_function)
arr = np.array([-1, 2, 5, 8])
print(f"\nVectorizare funcție custom:")
print(f"Input: {arr}")
print(f"Output: {vectorized_func(arr)}")

# 17.6 np.apply_along_axis
arr = np.random.rand(3, 4)
print(f"\nArray pentru apply_along_axis:\n{arr}")

# Aplicare funcție pe fiecare coloană
result = np.apply_along_axis(lambda x: x.max() - x.min(), 0, arr)
print(f"Range pe coloane: {result}")

# 17.7 Stride tricks (avansat - atenție la memorie!)
from numpy.lib.stride_tricks import as_strided

arr = np.arange(10)
# Creare ferestre glisante
shape = (len(arr) - 2, 3)  # 8 ferestre de lungime 3
strides = (arr.strides[0], arr.strides[0])
windows = as_strided(arr, shape=shape, strides=strides)
print(f"\nFerestre glisante (stride tricks):")
print(f"Original: {arr}")
print(f"Ferestre:\n{windows}")

# 17.8 np.pad (adăugare padding)
arr = np.array([[1, 2], [3, 4]])
print(f"\nArray original:\n{arr}")

padded = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
print(f"Cu padding (constant=0):\n{padded}")

padded_reflect = np.pad(arr, pad_width=1, mode='reflect')
print(f"Cu padding (reflect):\n{padded_reflect}")

# 17.9 np.clip (limitare valori)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
clipped = np.clip(arr, 3, 7)
print(f"\nClip între [3, 7]:")
print(f"Original: {arr}")
print(f"Clipped: {clipped}")

# 17.10 np.digitize (binning)
arr = np.array([0.2, 6.4, 3.0, 1.6, 9.2, 4.5])
bins = np.array([0, 3, 6, 9])
indices = np.digitize(arr, bins)
print(f"\nDigitize (binning):")
print(f"Valori: {arr}")
print(f"Bins: {bins}")
print(f"Indici: {indices}")

# 17.11 Structured arrays
dt = np.dtype([('nume', 'U10'), ('varsta', 'i4'), ('inaltime', 'f4')])
persoane = np.array([
    ('Ana', 25, 1.65),
    ('Ion', 30, 1.80),
    ('Maria', 22, 1.70)
], dtype=dt)

print(f"\nStructured array:")
print(persoane)
print(f"Nume: {persoane['nume']}")
print(f"Vârstă medie: {persoane['varsta'].mean()}")

# 17.12 Record arrays (acces ca atribute)
persoane_rec = persoane.view(np.recarray)
print(f"\nRecord array (acces ca atribute):")
print(f"Înălțime: {persoane_rec.inaltime}")

print("\n" + "="*80 + "\n")

# ============================================================================
# 18. CAZURI DE UTILIZARE PRACTICE
# ============================================================================

print("18. CAZURI DE UTILIZARE PRACTICE\n")

# 18.1 Procesare imagini (simulare)
print("Exemplu 1: Procesare imagine")
imagine = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
print(f"Imagine RGB shape: {imagine.shape}")

# Conversie la grayscale
grayscale = imagine.mean(axis=2).astype(np.uint8)
print(f"Grayscale shape: {grayscale.shape}")

# Aplicare filtru
filtru = np.array([[1, 1, 1],
                   [1, 2, 1],
                   [1, 1, 1]]) / 10
print(f"Filtru blur:\n{filtru}")

# 18.2 Normalizare date
print(f"\nExemplu 2: Normalizare date")
date = np.random.randn(100, 5) * 10 + 50
print(f"Date originale - medie: {date.mean():.2f}, std: {date.std():.2f}")

# Z-score normalizare
date_norm = (date - date.mean(axis=0)) / date.std(axis=0)
print(f"Date normalizate - medie: {date_norm.mean():.2f}, std: {date_norm.std():.2f}")

# Min-Max normalizare
date_minmax = (date - date.min(axis=0)) / (date.max(axis=0) - date.min(axis=0))
print(f"Date min-max - min: {date_minmax.min():.2f}, max: {date_minmax.max():.2f}")

# 18.3 Sliding window pentru time series
print(f"\nExemplu 3: Sliding window")
time_series = np.sin(np.linspace(0, 10, 100))
window_size = 5

# Medie mobilă
moving_avg = np.convolve(time_series, np.ones(window_size)/window_size, mode='valid')
print(f"Serie originală: {len(time_series)} puncte")
print(f"Medie mobilă: {len(moving_avg)} puncte")

# 18.4 One-hot encoding
print(f"\nExemplu 4: One-hot encoding")
categorii = np.array([0, 2, 1, 0, 3, 2])
n_categorii = 4

one_hot = np.eye(n_categorii)[categorii]
print(f"Categorii: {categorii}")
print(f"One-hot:\n{one_hot}")

# 18.5 Calcul distanțe (ex: distanță Euclidiană)
print(f"\nExemplu 5: Matrice distanțe")
puncte = np.random.rand(5, 2)  # 5 puncte în 2D
print(f"Puncte:\n{puncte}")

# Calcul distanțe între toate perechile
diff = puncte[:, np.newaxis, :] - puncte[np.newaxis, :, :]
distante = np.sqrt((diff**2).sum(axis=2))
print(f"Matrice distanțe:\n{distante}")

# 18.6 Agregare pe grupe
print(f"\nExemplu 6: Agregare pe grupe")
grupe = np.array([0, 0, 1, 1, 2, 2])
valori = np.array([10, 15, 20, 25, 30, 35])

for grup in np.unique(grupe):
    masca = grupe == grup
    print(f"Grup {grup}: suma={valori[masca].sum()}, medie={valori[masca].mean()}")

# 18.7 Generare batch-uri pentru ML
print(f"\nExemplu 7: Batch-uri pentru Machine Learning")
n_samples = 100
batch_size = 16

X = np.random.rand(n_samples, 10)  # Features
y = np.random.randint(0, 2, n_samples)  # Labels

n_batches = n_samples // batch_size
print(f"Total samples: {n_samples}")
print(f"Batch size: {batch_size}")
print(f"Number of batches: {n_batches}")

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X[start:end]
    y_batch = y[start:end]
    print(f"Batch {i}: X shape {X_batch.shape}, y shape {y_batch.shape}")
    if i >= 2:  # Afișăm doar primele 3 batch-uri
        print("...")
        break

# 18.8 Calcul cosinul similaritate
print(f"\nExemplu 8: Cosinus similaritate")
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])

cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Vector 1: {vec1}")
print(f"Vector 2: {vec2}")
print(f"Cosinus similaritate: {cosine_sim:.4f}")

print("\n" + "="*80 + "\n")

# ============================================================================
# 19. BEST PRACTICES ȘI ERORI COMUNE
# ============================================================================

print("19. BEST PRACTICES ȘI ERORI COMUNE\n")

print("BEST PRACTICES:")
print("-" * 80)
print("""
1. Folosește întotdeauna tipul de date corect pentru a economisi memorie
2. Evită copiile inutile - folosește view-uri când e posibil
3. Vectorizează operațiile - evită loop-urile Python
4. Folosește broadcasting în loc de tile/repeat
5. Profită de funcțiile NumPy optimizate
6. Verifică întotdeauna shape-ul arrays-urilor în debugging
7. Folosește axis parameter pentru operații pe dimensiuni specifice
8. Documentează shape-urile așteptate în funcții
9. Folosește assert pentru a verifica shape-uri
10. Setează seed pentru reproducibilitate în experimente

ERORI COMUNE:
""")
print("-" * 80)

# Eroare 1: Modificare accidentală prin view
print("\n1. ATENȚIE la view vs copy:")
arr = np.array([1, 2, 3, 4, 5])
slice_view = arr[1:4]
slice_view[0] = 999
print(f"Array original (modificat prin view!): {arr}")
print("Soluție: folosește .copy() când vrei independență")

# Eroare 2: Broadcasting neintentionat
print("\n2. Broadcasting neașteptat:")
try:
    a = np.array([[1, 2, 3]])  # (1, 3)
    b = np.array([[1], [2]])   # (2, 1)
    c = a + b  # Broadcasting la (2, 3)
    print(f"Shape neașteptat: {c.shape}")
    print("Verifică întotdeauna shape-urile!")
except Exception as e:
    print(f"Eroare: {e}")

# Eroare 3: Comparare cu ==
print("\n3. Comparare arrays (nu folosi ==):")
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
print(f"a == b returnează array: {a == b}")
print(f"Folosește np.array_equal: {np.array_equal(a, b)}")
print(f"Sau np.allclose pentru float: {np.allclose(a, b)}")

# Eroare 4: Diviziune prin zero
print("\n4. Diviziune prin zero:")
a = np.array([1, 2, 3])
b = np.array([0, 2, 0])
with np.errstate(divide='ignore', invalid='ignore'):
    result = a / b
    print(f"Rezultat: {result}")
    print(f"Folosește np.errstate sau verifică manual")

# Eroare 5: Flatten vs ravel
print("\n5. Flatten vs ravel:")
arr = np.array([[1, 2], [3, 4]])
flat = arr.flatten()  # Copie
rav = arr.ravel()     # View (de obicei)
rav[0] = 999
print(f"Array după modificare ravel: {arr}")
print(f"Flatten = copie, ravel = view (de obicei)")

# Eroare 6: Indexare negativă
print("\n6. Indexare negativă (diferă de Python):")
arr = np.array([1, 2, 3, 4, 5])
print(f"arr[-1:]: {arr[-1:]}")  # Ultimul element
print(f"arr[:-1]: {arr[:-1]}")  # Toate fără ultimul

# Eroare 7: Tip date implicit
print("\n7. Atenție la tipul de date implicit:")
arr_int = np.array([1, 2, 3])
arr_float = arr_int / 2
print(f"Tip original: {arr_int.dtype}")
print(f"Tip după diviziune: {arr_float.dtype}")

# Eroare 8: Axis confusion
print("\n8. Confuzie cu axis:")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array shape: {arr.shape}")
print(f"Sum axis=0 (pe coloane): {np.sum(arr, axis=0)}")
print(f"Sum axis=1 (pe linii): {np.sum(arr, axis=1)}")

print("\n" + "="*80 + "\n")

# ============================================================================
# 20. RESURSE ȘI CONCLUZIE
# ============================================================================

print("20. RESURSE ȘI URMĂTORII PAȘI\n")

print("""
DOCUMENTAȚIE OFICIALĂ:
- NumPy Docs: https://numpy.org/doc/
- NumPy User Guide: https://numpy.org/doc/stable/user/
- NumPy API Reference: https://numpy.org/doc/stable/reference/

TUTORIALE AVANSATE:
- NumPy for Absolute Beginners
- NumPy Tutorials (oficial)
- From Python to NumPy (book)

BIBLIOTECI COMPLEMENTARE:
- SciPy: funcții științifice avansate
- Pandas: manipulare date tabulare
- Matplotlib: vizualizare
- Scikit-learn: machine learning
- TensorFlow/PyTorch: deep learning

PRACTICĂ:
1. Rezolvă probleme pe HackerRank/LeetCode
2. Contribuie la proiecte open-source
3. Implementează algoritmi folosind NumPy
4. Participă la competiții Kaggle
5. Citește cod din biblioteci populare

SFATURI FINALE:
- Citește documentația - este excelentă!
- Experimentează în Jupyter notebooks
- Măsoară performanța cu %%timeit
- Înțelege broadcasting - este esențial
- Gândește vectorizat, nu în loop-uri
- NumPy este fundația ecosistemului științific Python

================================================================================
                            SUCCES ÎN ÎNVĂȚARE!
================================================================================

Acest fișier conține toate conceptele fundamentale și avansate despre NumPy.
Revizuiește secțiunile în funcție de nevoile tale și practică exemplele.
NumPy este o unealtă puternică - cu practică vei deveni expert!

Autor: Ghid Complet NumPy
Data: 2025
Versiune: Completă și actualizată
""")

print("="*80)
print("FINAL - Ghid Complet NumPy")
print("="*80)