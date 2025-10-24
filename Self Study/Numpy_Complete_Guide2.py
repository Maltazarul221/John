# ============================================================================
# 21. BROADCASTING DETALIAT ȘI AVANSAT
# ============================================================================
import numpy as np

print("21. BROADCASTING DETALIAT ȘI AVANSAT\n")

"""
Broadcasting permite operații între arrays de forme diferite fără copiere explicită.
Este una dintre cele mai puternice caracteristici ale NumPy.
"""

print("REGULILE BROADCASTING:")
print("""
1. Dacă arrays au număr diferit de dimensiuni, forma celui mai mic 
   este completată cu 1 în stânga.
2. Arrays sunt compatibile pe o dimensiune dacă:
   - Au aceeași dimensiune, SAU
   - Una din dimensiuni este 1
3. După broadcasting, fiecare array se comportă ca și cum ar avea 
   forma maximă de-a lungul fiecărei dimensiuni.
""")

# 21.1 Broadcasting 1D + scalar
print("\n21.1 Broadcasting 1D + scalar:")
a = np.array([1, 2, 3])
b = 10
print(f"a shape: {a.shape}")
print(f"b (scalar)")
result = a + b  # b devine [10, 10, 10]
print(f"Result: {result}")

# 21.2 Broadcasting 1D + 1D
print("\n21.2 Broadcasting 1D + 1D (trebuie dimensiuni compatibile):")
a = np.array([1, 2, 3])  # (3,)
b = np.array([10, 20, 30])  # (3,)
print(f"a + b: {a + b}")

# Pentru broadcasting real, trebuie să facem unul (3,1) și altul (3,)
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])  # (3,)
print(f"\na shape: {a.shape}, b shape: {b.shape}")
result = a + b  # Devine (3, 3)
print(f"Result shape: {result.shape}")
print(f"Result:\n{result}")

# 21.3 Broadcasting 2D + 1D
print("\n21.3 Broadcasting 2D + 1D:")
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # (3, 3)
vector = np.array([10, 20, 30])  # (3,)

print(f"Matrix shape: {matrix.shape}")
print(f"Vector shape: {vector.shape}")
result = matrix + vector  # vector este broadcast pe fiecare rând
print(f"Result:\n{result}")

# Broadcast pe coloane (trebuie reshape)
vector_col = np.array([[10], [20], [30]])  # (3, 1)
result_col = matrix + vector_col
print(f"\nBroadcast pe coloane:\n{result_col}")

# 21.4 Broadcasting complex 3D
print("\n21.4 Broadcasting 3D:")
a = np.ones((3, 1, 4))  # (3, 1, 4)
b = np.ones((1, 2, 4))  # (1, 2, 4)
result = a + b  # Rezultat (3, 2, 4)
print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"Result shape: {result.shape}")

# 21.5 Broadcasting pentru operații outer
print("\n21.5 Outer product cu broadcasting:")
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30])

# Outer product manual
outer = x[:, np.newaxis] * y[np.newaxis, :]
print(f"Outer product:\n{outer}")

# Sau
outer2 = x.reshape(-1, 1) * y.reshape(1, -1)
print(f"Outer product (reshape):\n{outer2}")

# 21.6 Broadcasting pentru distanțe
print("\n21.6 Calculul distanțelor folosind broadcasting:")
points = np.random.rand(5, 2)  # 5 puncte în 2D
print(f"Puncte:\n{points}")

# Calculăm distanțele între toate perechile
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (5, 5, 2)
distances = np.sqrt((diff ** 2).sum(axis=2))  # (5, 5)
print(f"\nMatrice distanțe:\n{distances}")

# 21.7 Broadcasting pentru normalizare
print("\n21.7 Normalizare per rând cu broadcasting:")
data = np.random.rand(4, 5) * 100
print(f"Date originale:\n{data}")

# Normalizare Z-score per rând
mean = data.mean(axis=1, keepdims=True)  # (4, 1)
std = data.std(axis=1, keepdims=True)  # (4, 1)
normalized = (data - mean) / std
print(f"\nNormalizat (Z-score):\n{normalized}")

# 21.8 Erori comune de broadcasting
print("\n21.8 Erori comune de broadcasting:")
print("Exemplu eroare:")
try:
    a = np.ones((3, 4))
    b = np.ones((3,))
    result = a + b  # Funcționează - (3, 4) + (3,) -> broadcast pe ultima dimensiune
    print("Success - b este broadcast pe coloane")

    b = np.ones((4, 3))
    result = a + b  # EROARE - incompatibile
except ValueError as e:
    print(f"Eroare: {e}")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 22. MEMORY LAYOUT ȘI STRIDES
# ============================================================================

print("22. MEMORY LAYOUT ȘI STRIDES\n")

"""
Înțelegerea layout-ului memoriei este crucială pentru optimizarea performanței.
NumPy stochează arrays în memorie continuă, dar ordinea poate varia.
"""

# 22.1 C-order vs Fortran-order
print("22.1 C-order (row-major) vs Fortran-order (column-major):")

# C-order (default) - row-major
arr_c = np.array([[1, 2, 3],
                  [4, 5, 6]], order='C')
print("C-order (row-major):")
print(f"Array:\n{arr_c}")
print(f"Flags:\n{arr_c.flags}")
print(f"Strides: {arr_c.strides}")

# Fortran-order - column-major
arr_f = np.array([[1, 2, 3],
                  [4, 5, 6]], order='F')
print("\nFortran-order (column-major):")
print(f"Array:\n{arr_f}")
print(f"Flags:\n{arr_f.flags}")
print(f"Strides: {arr_f.strides}")

# 22.2 Strides explicație
print("\n22.2 Strides - pași în memorie:")
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]], dtype=np.int32)
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")
print(f"Strides: {arr.strides}")
print(f"Itemsize: {arr.itemsize} bytes")
print("\nExplicație:")
print(f"Pentru a merge la următoarea coloană: {arr.strides[1]} bytes")
print(f"Pentru a merge la următorul rând: {arr.strides[0]} bytes")

# 22.3 View vs Copy
print("\n22.3 View vs Copy:")
original = np.arange(12).reshape(3, 4)
print(f"Original:\n{original}")

# View (shared memory)
view = original[1:3, 1:3]
print(f"\nView:\n{view}")
print(f"View bazat pe original: {view.base is original}")

view[0, 0] = 999
print(f"\nDupă modificare view:")
print(f"View:\n{view}")
print(f"Original (modificat!):\n{original}")

# Copy (new memory)
original = np.arange(12).reshape(3, 4)
copy = original[1:3, 1:3].copy()
print(f"\nCopy:\n{copy}")
print(f"Copy bazat pe original: {copy.base is original}")

copy[0, 0] = 999
print(f"\nDupă modificare copy:")
print(f"Copy:\n{copy}")
print(f"Original (nemodificat):\n{original}")

# 22.4 Contiguous arrays
print("\n22.4 Contiguous arrays:")
arr = np.arange(12).reshape(3, 4)
print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")
print(f"F-contiguous: {arr.flags['F_CONTIGUOUS']}")

# Slicing poate face array non-contiguous
sliced = arr[::2, ::2]
print(f"\nSliced array:")
print(f"C-contiguous: {sliced.flags['C_CONTIGUOUS']}")
print(f"F-contiguous: {sliced.flags['F_CONTIGUOUS']}")

# Forțare contiguitate
contiguous = np.ascontiguousarray(sliced)
print(f"\nDupă ascontiguousarray:")
print(f"C-contiguous: {contiguous.flags['C_CONTIGUOUS']}")

# 22.5 Performanță - contiguous vs non-contiguous
print("\n22.5 Performanță - contiguous vs non-contiguous:")
large_arr = np.random.rand(1000, 1000)
contiguous = np.ascontiguousarray(large_arr)
non_contiguous = large_arr[::2, ::2]

import time

# Operație pe contiguous
start = time.time()
result1 = contiguous.sum()
time_cont = time.time() - start

# Operație pe non-contiguous
start = time.time()
result2 = non_contiguous.sum()
time_non_cont = time.time() - start

print(f"Timp contiguous: {time_cont:.6f}s")
print(f"Timp non-contiguous: {time_non_cont:.6f}s")
print(f"Ratio: {time_non_cont / time_cont:.2f}x")

# 22.6 As_strided - avansat (atenție!)
print("\n22.6 as_strided - manipulare avansată:")
from numpy.lib.stride_tricks import as_strided

arr = np.arange(10)
print(f"Array original: {arr}")

# Creare ferestre glisante fără copiere
shape = (8, 3)  # 8 ferestre de 3 elemente
strides = (arr.strides[0], arr.strides[0])
windows = as_strided(arr, shape=shape, strides=strides)
print(f"Ferestre glisante:\n{windows}")
print("ATENȚIE: as_strided este periculos - poți accesa memorie invalidă!")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 23. STRUCTURED ARRAYS ȘI RECORD ARRAYS
# ============================================================================

print("23. STRUCTURED ARRAYS ȘI RECORD ARRAYS\n")

"""
Structured arrays permit tipuri de date eterogene în același array,
similar cu tabele sau structuri C.
"""

# 23.1 Creare structured array
print("23.1 Creare structured array:")
dt = np.dtype([('nume', 'U20'), ('vârstă', 'i4'), ('înălțime', 'f4')])
persoane = np.array([
    ('Ana Pop', 25, 1.65),
    ('Ion Ionescu', 30, 1.80),
    ('Maria Popescu', 22, 1.70),
    ('Radu Vasile', 35, 1.75)
], dtype=dt)

print(f"Persoane:\n{persoane}")
print(f"Dtype: {persoane.dtype}")

# 23.2 Accesare câmpuri
print("\n23.2 Accesare câmpuri:")
print(f"Nume: {persoane['nume']}")
print(f"Vârste: {persoane['vârstă']}")
print(f"Înălțimi: {persoane['înălțime']}")

# 23.3 Filtrare
print("\n23.3 Filtrare:")
tineri = persoane[persoane['vârstă'] < 30]
print(f"Persoane sub 30 ani:\n{tineri}")

# 23.4 Sortare după câmp
print("\n23.4 Sortare după câmp:")
sortate_vârstă = np.sort(persoane, order='vârstă')
print(f"Sortate după vârstă:\n{sortate_vârstă}")

# 23.5 Record arrays (acces ca atribute)
print("\n23.5 Record arrays:")
rec_arr = persoane.view(np.recarray)
print(f"Nume (ca atribut): {rec_arr.nume}")
print(f"Vârste (ca atribut): {rec_arr.vârstă}")

# 23.6 Creare din dicționar de arrays
print("\n23.6 Creare din dicționar:")
data = {
    'produse': ['Laptop', 'Mouse', 'Tastatură'],
    'preț': [3000, 50, 150],
    'stoc': [15, 100, 50]
}

# Conversie la structured array
dt = np.dtype([('produse', 'U20'), ('preț', 'i4'), ('stoc', 'i4')])
struct_arr = np.array(list(zip(data['produse'], data['preț'], data['stoc'])), dtype=dt)
print(f"Structured array:\n{struct_arr}")

# 23.7 Nested structured types
print("\n23.7 Nested structured types:")
dt_nested = np.dtype([
    ('angajat', [('nume', 'U20'), ('id', 'i4')]),
    ('salariu', 'f4'),
    ('departament', 'U20')
])

angajați = np.array([
    (('Ana Pop', 101), 3000.0, 'IT'),
    (('Ion Ionescu', 102), 4500.0, 'HR')
], dtype=dt_nested)

print(f"Angajați:\n{angajați}")
print(f"Nume: {angajați['angajat']['nume']}")
print(f"ID: {angajați['angajat']['id']}")

# 23.8 Operații pe structured arrays
print("\n23.8 Operații pe structured arrays:")
print(f"Medie vârstă: {persoane['vârstă'].mean()}")
print(f"Max înălțime: {persoane['înălțime'].max()}")
print(f"Persoane over 1.70m: {(persoane['înălțime'] > 1.70).sum()}")

# 23.9 Conversie la DataFrame (dacă pandas e disponibil)
print("\n23.9 Interoperabilitate:")
print("Structured arrays pot fi convertite ușor în Pandas DataFrame:")
print("df = pd.DataFrame(persoane)")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 24. UNIVERSAL FUNCTIONS (UFUNCS) AVANSAT
# ============================================================================

print("24. UNIVERSAL FUNCTIONS (UFUNCS) AVANSAT\n")

"""
Ufuncs sunt funcții vectorizate care operează element-wise și sunt 
implementate în C pentru performanță maximă.
"""

# 24.1 Metode ufunc
print("24.1 Metode ufunc:")
arr = np.array([1, 2, 3, 4, 5])

# reduce - aplică operația cumulativ
print(f"Array: {arr}")
print(f"np.add.reduce (sumă): {np.add.reduce(arr)}")
print(f"np.multiply.reduce (produs): {np.multiply.reduce(arr)}")

# accumulate - reduce cumulativ
print(f"\nnp.add.accumulate: {np.add.accumulate(arr)}")
print(f"np.multiply.accumulate: {np.multiply.accumulate(arr)}")

# reduceat - reduce pe slice-uri
indices = [0, 2, 4]
print(f"\nnp.add.reduceat cu indices {indices}: {np.add.reduceat(arr, indices)}")

# outer - produs outer
arr1 = np.array([1, 2, 3])
arr2 = np.array([10, 20])
print(f"\nnp.multiply.outer:\n{np.multiply.outer(arr1, arr2)}")

# 24.2 Creare ufunc proprie
print("\n24.2 Creare ufunc proprie cu frompyfunc:")


def my_function(x, y):
    """Funcție Python simplă"""
    return x + 2 * y


# Conversie la ufunc
my_ufunc = np.frompyfunc(my_function, 2, 1)  # 2 inputs, 1 output

x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
result = my_ufunc(x, y)
print(f"x: {x}")
print(f"y: {y}")
print(f"my_ufunc(x, y): {result}")

# 24.3 Vectorize (mai flexibil)
print("\n24.3 np.vectorize:")


def complex_function(x):
    if x < 0:
        return 0
    elif x < 5:
        return x * 2
    else:
        return x ** 2


vec_func = np.vectorize(complex_function)
arr = np.array([-2, 1, 3, 7, 10])
print(f"Input: {arr}")
print(f"Output: {vec_func(arr)}")

# 24.4 At method - operații in-place pe indici
print("\n24.4 At method - operații selective:")
arr = np.array([1, 2, 3, 4, 5])
indices = [0, 2, 4]
print(f"Array original: {arr}")

np.add.at(arr, indices, 10)
print(f"După add.at (indices {indices}, value 10): {arr}")

# 24.5 Ufuncs custom cu numba
print("\n24.5 Ufuncs cu numba (high performance):")
numba_example = """
from numba import vectorize

@vectorize
def my_fast_func(x, y):
    return x + 2*y + x*y

# Folosire
result = my_fast_func(x_array, y_array)

# Mult mai rapid decât np.frompyfunc!
"""
print(numba_example)

# 24.6 Broadcasting cu ufuncs
print("\n24.6 Broadcasting automată cu ufuncs:")
x = np.array([[1, 2, 3],
              [4, 5, 6]])
y = np.array([10, 20, 30])

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
result = np.add(x, y)  # Broadcasting automată
print(f"np.add(x, y):\n{result}")

# 24.7 Output parameter
print("\n24.7 Output parameter (evită alocare):")
x = np.array([1, 2, 3, 4, 5])
output = np.empty_like(x)

np.multiply(x, 2, out=output)
print(f"Output (x * 2): {output}")

# 24.8 Where parameter
print("\n24.8 Where parameter (operații condiționate):")
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
condition = x > 2

result = np.add(x, y, where=condition)
print(f"x: {x}")
print(f"y: {y}")
print(f"condition (x > 2): {condition}")
print(f"np.add(x, y, where=condition): {result}")

# 24.9 Casting rules
print("\n24.9 Casting rules:")
x_int = np.array([1, 2, 3], dtype=np.int32)
y_float = np.array([1.5, 2.5, 3.5], dtype=np.float64)

result = np.add(x_int, y_float)
print(f"x dtype: {x_int.dtype}")
print(f"y dtype: {y_float.dtype}")
print(f"result dtype: {result.dtype}")
print("NumPy face upcasting automat la tipul mai general")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 25. POLYNOMIAL OPERATIONS
# ============================================================================

print("25. POLYNOMIAL OPERATIONS\n")

"""
NumPy oferă suport comprehensiv pentru operații cu polinoame.
"""

# 25.1 Creare și evaluare polinoame
print("25.1 Creare și evaluare polinoame:")

# Polinom: 2x^2 + 3x + 1
coeffs = [2, 3, 1]  # De la puterea cea mai mare
p = np.poly1d(coeffs)
print(f"Polinom: {p}")

# Evaluare
x = 5
result = p(x)
print(f"p({x}) = {result}")

# Evaluare pe array
x_arr = np.array([1, 2, 3, 4, 5])
results = p(x_arr)
print(f"p({x_arr}) = {results}")

# 25.2 Operații aritmetice cu polinoame
print("\n25.2 Operații aritmetice:")
p1 = np.poly1d([1, 2])  # x + 2
p2 = np.poly1d([1, -3])  # x - 3

print(f"p1: {p1}")
print(f"p2: {p2}")
print(f"p1 + p2: {p1 + p2}")
print(f"p1 * p2: {p1 * p2}")
print(f"p1 - p2: {p1 - p2}")

# 25.3 Derivate și integrale
print("\n25.3 Derivate și integrale:")
p = np.poly1d([1, 2, 3, 4])  # x^3 + 2x^2 + 3x + 4
print(f"Polinom original: {p}")

deriv = p.deriv()
print(f"Derivată: {deriv}")

deriv2 = p.deriv(2)
print(f"Derivată a doua: {deriv2}")

integral = p.integ()
print(f"Integrală (C=0): {integral}")

integral_c = p.integ(k=5)  # cu constantă
print(f"Integrală (C=5): {integral_c}")

# 25.4 Rădăcini
print("\n25.4 Găsire rădăcini:")
p = np.poly1d([1, -3, 2])  # x^2 - 3x + 2 = (x-1)(x-2)
roots = p.roots
print(f"Polinom: {p}")
print(f"Rădăcini: {roots}")

# Verificare
print(f"p({roots[0]}) = {p(roots[0])}")
print(f"p({roots[1]}) = {p(roots[1])}")

# 25.5 Construire polinom din rădăcini
print("\n25.5 Construire polinom din rădăcini:")
roots = [1, 2, 3]
p = np.poly(roots)
print(f"Rădăcini: {roots}")
print(f"Polinom: {np.poly1d(p)}")

# 25.6 Fitting polinomal
print("\n25.6 Polynomial fitting:")
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 3, 7, 13, 21, 31])  # Aproximativ y = x^2 + 1

# Fit grad 2
coeffs = np.polyfit(x, y, 2)
p = np.poly1d(coeffs)
print(f"Date x: {x}")
print(f"Date y: {y}")
print(f"Polinom fitted (grad 2): {p}")

# Evaluare fitted polynomial
y_pred = p(x)
print(f"Predicții: {y_pred}")
print(f"Eroare: {np.abs(y - y_pred)}")

# 25.7 Evaluare eficientă - polyval
print("\n25.7 polyval - evaluare eficientă:")
coeffs = [1, 2, 3]  # x^2 + 2x + 3
x = np.array([1, 2, 3, 4, 5])

result = np.polyval(coeffs, x)
print(f"Coeficienți: {coeffs}")
print(f"x: {x}")
print(f"polyval(coeffs, x): {result}")

# 25.8 Polinoame Chebyshev, Legendre, etc.
print("\n25.8 Polinoame speciale:")
special_poly_example = """
# Polinoame Chebyshev
from numpy.polynomial import chebyshev as C
c = C.Chebyshev([1, 2, 3])

# Polinoame Legendre
from numpy.polynomial import legendre as L
l = L.Legendre([1, 2, 3])

# Polinoame Laguerre
from numpy.polynomial import laguerre as La
lag = La.Laguerre([1, 2, 3])

# Polinoame Hermite
from numpy.polynomial import hermite as H
h = H.Hermite([1, 2, 3])
"""
print(special_poly_example)

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 26. FFT (FAST FOURIER TRANSFORM)
# ============================================================================

print("26. FFT (FAST FOURIER TRANSFORM)\n")

"""
FFT este esențială pentru procesarea semnalelor și analiza frecvențelor.
"""

# 26.1 FFT de bază
print("26.1 FFT de bază:")

# Creare semnal simplu - sinusoide
t = np.linspace(0, 1, 500)
freq1, freq2 = 5, 10  # Hz
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

print(f"Semnal cu {len(signal)} sample-uri")
print(f"Frecvențe: {freq1} Hz și {freq2} Hz")

# FFT
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])

print(f"FFT result shape: {fft_result.shape}")
print(f"Frequencies shape: {frequencies.shape}")

# Magnitudine
magnitude = np.abs(fft_result)
print(f"Peak frequencies (primele 5): {frequencies[np.argsort(magnitude)[-5:]]}")

# 26.2 IFFT - Inverse FFT
print("\n26.2 Inverse FFT:")
reconstructed = np.fft.ifft(fft_result)
print(f"Semnal original == reconstructed: {np.allclose(signal, reconstructed.real)}")

# 26.3 FFT 2D (pentru imagini)
print("\n26.3 FFT 2D:")
image = np.random.rand(64, 64)
fft2d = np.fft.fft2(image)
print(f"Image shape: {image.shape}")
print(f"FFT2D shape: {fft2d.shape}")

# Shift pentru a centra frecvențele joase
fft2d_shifted = np.fft.fftshift(fft2d)
print("fftshift mută frecvențele joase în centru")

# 26.4 Real FFT (pentru semnale reale)
print("\n26.4 Real FFT (mai eficient pentru semnale reale):")
rfft_result = np.fft.rfft(signal)
print(f"FFT complex length: {len(fft_result)}")
print(f"RFFT length (jumătate + 1): {len(rfft_result)}")
print("RFFT returnează doar frecvențele pozitive (simetrie)")

# 26.5 Windowing
print("\n26.5 Windowing pentru reducerea leakage:")
from numpy.fft import fft

# Fără window
fft_no_window = fft(signal)

# Cu Hanning window
window = np.hanning(len(signal))
fft_windowed = fft(signal * window)

print(f"Signal fără window - peak: {np.max(np.abs(fft_no_window))}")
print(f"Signal cu Hanning window - peak: {np.max(np.abs(fft_windowed))}")

# 26.6 Filtrare în domeniul frecvenței
print("\n26.6 Filtrare în domeniul frecvenței:")

# Low-pass filter
fft_filtered = fft_result.copy()
cutoff_freq = 7  # Hz
cutoff_index = int(cutoff_freq * len(signal) / (1 / (t[1] - t[0])))
fft_filtered[cutoff_index:-cutoff_index] = 0

# Reconstruct
filtered_signal = np.fft.ifft(fft_filtered).real
print(f"Filtru low-pass la {cutoff_freq} Hz aplicat")
print(f"Signal energy reduced: {(signal ** 2).sum()} -> {(filtered_signal ** 2).sum()}")

# 26.7 Spectrograma (pentru semnale variabile în timp)
print("\n26.7 Spectrogramă (STFT - Short-Time Fourier Transform):")
spectrogram_example = """
from scipy import signal as sig

# STFT
f, t, Zxx = sig.stft(signal, fs=sampling_rate)

# Spectrogramă
plt.pcolormesh(t, f, np.abs(Zxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
"""
print(spectrogram_example)

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 27. MASKED ARRAYS
# ============================================================================

print("27. MASKED ARRAYS - LUCRU CU DATE INVALIDE\n")

"""
Masked arrays permit marcarea unor valori ca invalide fără a le șterge.
Util pentru date cu găuri, outliers, sau valori nedisponibile.
"""

# 27.1 Creare masked array
print("27.1 Creare masked array:")
data = np.array([1, 2, -999, 4, 5, -999, 7, 8])
mask = (data == -999)  # True pentru valori invalide

masked_arr = np.ma.masked_array(data, mask=mask)
print(f"Date originale: {data}")
print(f"Mască: {mask}")
print(f"Masked array: {masked_arr}")

# 27.2 Operații cu masked arrays
print("\n27.2 Operații (ignoră valorile masked):")
print(f"Sumă: {masked_arr.sum()}")
print(f"Medie: {masked_arr.mean()}")
print(f"Std: {masked_arr.std()}")
print(f"Min/Max: {masked_arr.min()} / {masked_arr.max()}")

# 27.3 Masked array din condiție
print("\n27.3 Masked array din condiție:")
data = np.array([1, 2, 100, 4, 5, 200, 7, 8])
masked = np.ma.masked_greater(data, 50)
print(f"Date: {data}")
print(f"Masked (>50): {masked}")

# Alte funcții de masking
masked_less = np.ma.masked_less(data, 5)
print(f"Masked (<5): {masked_less}")

masked_between = np.ma.masked_inside(data, 3, 7)
print(f"Masked (între 3 și 7): {masked_between}")

masked_outside = np.ma.masked_outside(data, 3, 7)
print(f"Masked (în afara 3-7): {masked_outside}")

# 27.4 Filled values
print("\n27.4 Filled values:")
masked = np.ma.masked_array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 1])
print(f"Masked array: {masked}")
print(f"Filled cu 0: {masked.filled(0)}")
print(f"Filled cu medie: {masked.filled(masked.mean())}")

# 27.5 Modificare mască
print("\n27.5 Modificare mască:")
arr = np.ma.array([1, 2, 3, 4, 5])
print(f"Array original: {arr}")

arr[2] = np.ma.masked
print(f"După masking element 2: {arr}")

# Unmask
arr[2] = 10
print(f"După unmask și setare: {arr}")

# 27.6 Combinare măști
print("\n27.6 Combinare măști:")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
mask1 = data < 3
mask2 = data > 6

combined_mask = mask1 | mask2  # OR
masked = np.ma.masked_array(data, mask=combined_mask)
print(f"Date: {data}")
print(f"Mască1 (<3): {mask1}")
print(f"Mască2 (>6): {mask2}")
print(f"Mască combinată (OR): {combined_mask}")
print(f"Masked array: {masked}")

# 27.7 Compressed (obține doar valorile valide)
print("\n27.7 Compressed:")
masked = np.ma.masked_array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 1])
compressed = masked.compressed()
print(f"Masked array: {masked}")
print(f"Compressed (doar valide): {compressed}")

# 27.8 Hardmask vs Softmask
print("\n27.8 Hardmask vs Softmask:")
soft = np.ma.array([1, 2, 3], mask=[0, 1, 0])
soft[1] = 999  # Poate suprascrie masked values
print(f"Softmask după modificare: {soft}")

hard = np.ma.array([1, 2, 3], mask=[0, 1, 0], hard_mask=True)
hard[1] = 999  # NU poate suprascrie
print(f"Hardmask după modificare: {hard}")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 28. MATRIX DECOMPOSITIONS AVANSATE
# ============================================================================

print("28. MATRIX DECOMPOSITIONS AVANSATE\n")

"""
Descompunerile matriceale sunt fundamentale în algebra liniară și ML.
"""

# 28.1 SVD (Singular Value Decomposition)
print("28.1 SVD - Singular Value Decomposition:")
A = np.random.rand(5, 3)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print(f"Matrix A shape: {A.shape}")
print(f"U shape: {U.shape}")
print(f"s shape: {s.shape}")
print(f"Vt shape: {Vt.shape}")

# Reconstruct
S = np.diag(s)
reconstructed = U @ S @ Vt
print(f"A == U @ S @ Vt: {np.allclose(A, reconstructed)}")

# 28.2 Eigenvalue Decomposition
print("\n28.2 Eigenvalue Decomposition:")
A = np.array([[1, 2],
              [2, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Matrix A:\n{A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verificare: A @ v = λ @ v
v = eigenvectors[:, 0]
λ = eigenvalues[0]
print(f"\nVerificare A @ v = λ @ v:")
print(f"A @ v = {A @ v}")
print(f"λ @ v = {λ * v}")

# 28.3 QR Decomposition
print("\n28.3 QR Decomposition:")
A = np.random.rand(5, 3)
Q, R = np.linalg.qr(A)

print(f"Matrix A shape: {A.shape}")
print(f"Q shape (orthogonal): {Q.shape}")
print(f"R shape (upper triangular): {R.shape}")
print(f"Q @ R == A: {np.allclose(Q @ R, A)}")
print(f"Q^T @ Q == I: {np.allclose(Q.T @ Q[:, :3], np.eye(3))}")

# 28.4 Cholesky Decomposition
print("\n28.4 Cholesky Decomposition (doar pentru pozitiv definite):")
# Creăm o matrice pozitiv definită
A = np.array([[4, 2],
              [2, 3]])
L = np.linalg.cholesky(A)

print(f"Matrix A (pozitiv definită):\n{A}")
print(f"L (lower triangular):\n{L}")
print(f"L @ L.T == A: {np.allclose(L @ L.T, A)}")

# 28.5 LU Decomposition (necesită scipy)
print("\n28.5 LU Decomposition:")
lu_example = """
from scipy.linalg import lu

A = np.array([[2, 5, 8, 7],
              [5, 2, 2, 8],
              [7, 5, 6, 6],
              [5, 4, 4, 8]])

P, L, U = lu(A)  # PA = LU

# P = permutation matrix
# L = lower triangular
# U = upper triangular
"""
print(lu_example)

# 28.6 Aplicații SVD - PCA
print("\n28.6 Aplicații SVD - PCA (Principal Component Analysis):")
# Date 2D
X = np.random.randn(100, 5)  # 100 sample, 5 features

# Centrare
X_centered = X - X.mean(axis=0)

# SVD
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Prima componentă principală
pc1 = Vt[0]
print(f"Prima componentă principală: {pc1}")

# Variance explained
variance_explained = (s ** 2) / np.sum(s ** 2)
print(f"Variance explained: {variance_explained}")

# Proiecție pe primele 2 componente
X_reduced = X_centered @ Vt[:2].T
print(f"Date reduse shape: {X_reduced.shape}")

# 28.7 Aplicații - compresie imagine
print("\n28.7 Aplicații SVD - compresie imagine:")
compression_example = """
# Imagine grayscale
image = np.random.rand(512, 512)

# SVD
U, s, Vt = np.linalg.svd(image, full_matrices=False)

# Păstrăm doar primele k valori singulare
k = 50
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Imagine comprimată
compressed = U_k @ np.diag(s_k) @ Vt_k

# Raport compresie
original_size = image.size
compressed_size = U_k.size + s_k.size + Vt_k.size
compression_ratio = original_size / compressed_size

print(f'Compression ratio: {compression_ratio:.2f}x')
"""
print(compression_example)

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 29. NUMERICAL INTEGRATION
# ============================================================================

print("29. NUMERICAL INTEGRATION\n")

"""
NumPy oferă metode de bază pentru integrare numerică.
Pentru funcționalitate avansată, folosește scipy.integrate.
"""

# 29.1 Trapezoid rule
print("29.1 Trapezoid rule:")
x = np.linspace(0, np.pi, 100)
y = np.sin(x)

integral = np.trapz(y, x)
print(f"Integrală sin(x) de la 0 la π: {integral}")
print(f"Valoare exactă: {2.0}")
print(f"Eroare: {abs(integral - 2.0):.6f}")

# 29.2 Cumulative integration
print("\n29.2 Integrare cumulativă:")
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Integrală cumulativă cu trapz
cumulative = np.array([np.trapz(y[:i + 1], x[:i + 1]) for i in range(len(x))])
print(f"Integrală cumulativă shape: {cumulative.shape}")
print(f"Valoare finală: {cumulative[-1]}")

# 29.3 Simpson's rule (scipy)
print("\n29.3 Simpson's rule:")
simpson_example = """
from scipy.integrate import simps

x = np.linspace(0, np.pi, 100)
y = np.sin(x)

integral = simps(y, x)
print(f'Integral: {integral}')

# Mai precis decât trapz pentru funcții smooth
"""
print(simpson_example)

# 29.4 Integrare 2D
print("\n29.4 Integrare 2D:")
integration_2d_example = """
from scipy.integrate import dblquad

# Integrare f(x, y) = x*y pe [0,1] x [0,1]
def f(y, x):
    return x * y

result, error = dblquad(f, 0, 1, 0, 1)
print(f'Integral: {result}')
print(f'Error estimate: {error}')
"""
print(integration_2d_example)

# 29.5 Aplicație - area under curve
print("\n29.5 Aplicație - Area Under Curve (AUC):")
# Simulare ROC curve
fpr = np.array([0, 0.1, 0.2, 0.4, 0.7, 1.0])
tpr = np.array([0, 0.3, 0.6, 0.8, 0.95, 1.0])

auc = np.trapz(tpr, fpr)
print(f"False Positive Rate: {fpr}")
print(f"True Positive Rate: {tpr}")
print(f"AUC (Area Under ROC Curve): {auc:.3f}")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 30. INTERFACING CU C/FORTRAN
# ============================================================================

print("30. INTERFACING CU C/FORTRAN\n")

"""
Pentru performanță maximă, poți folosi cod C/Fortran cu NumPy.
"""

print("30.1 Ctypes:")
ctypes_example = """
import ctypes
import numpy as np

# Încarcă bibliotecă C
lib = ctypes.CDLL('./mylib.so')  # Linux
# lib = ctypes.CDLL('./mylib.dll')  # Windows

# Specifică tipurile funcției
lib.my_function.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_int
]
lib.my_function.restype = ctypes.c_double

# Folosire
arr = np.array([1.0, 2.0, 3.0, 4.0])
result = lib.my_function(arr, len(arr))
"""
print(ctypes_example)

print("\n30.2 F2PY (Fortran to Python):")
f2py_example = """
! Cod Fortran (mymodule.f90)
subroutine compute_sum(arr, n, result)
    integer, intent(in) :: n
    real*8, intent(in) :: arr(n)
    real*8, intent(out) :: result
    integer :: i

    result = 0.0
    do i = 1, n
        result = result + arr(i)
    end do
end subroutine

# Compilare
# f2py -c mymodule.f90 -m mymodule

# Folosire în Python
import mymodule
arr = np.array([1.0, 2.0, 3.0, 4.0])
result = mymodule.compute_sum(arr)
"""
print(f2py_example)

print("\n30.3 Cython:")
cython_example = """
# mymodule.pyx
import numpy as np
cimport numpy as np

def fast_sum(np.ndarray[np.float64_t, ndim=1] arr):
    cdef int i
    cdef int n = arr.shape[0]
    cdef double total = 0.0

    for i in range(n):
        total += arr[i]

    return total

# Compilare: python setup.py build_ext --inplace
# Folosire: from mymodule import fast_sum
"""
print(cython_example)

print("\n30.4 Numba (cel mai simplu):")
numba_example = """
from numba import jit
import numpy as np

@jit(nopython=True)
def fast_function(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] ** 2
    return total

arr = np.random.rand(1000000)
result = fast_function(arr)  # Compilat la prima rulare, rapid după

# Mult mai rapid decât Python pur!
# Aproape la fel de rapid ca C!
"""
print(numba_example)

print("\n30.5 Performance comparison:")
performance_example = """
import numpy as np
import time

arr = np.random.rand(1000000)

# Python pur
start = time.time()
total = sum(x**2 for x in arr)
time_python = time.time() - start

# NumPy
start = time.time()
total = np.sum(arr ** 2)
time_numpy = time.time() - start

# Numba
@jit(nopython=True)
def numba_sum(arr):
    total = 0.0
    for x in arr:
        total += x ** 2
    return total

start = time.time()
total = numba_sum(arr)
time_numba = time.time() - start

print(f'Python: {time_python:.4f}s')
print(f'NumPy: {time_numpy:.4f}s ({time_python/time_numpy:.0f}x faster)')
print(f'Numba: {time_numba:.4f}s ({time_python/time_numba:.0f}x faster)')
"""
print(performance_example)

print("\n" + "=" * 80 + "\n")

# ============================================================================
# CONCLUZIE FINALĂ EXTINSĂ
# ============================================================================

print("CONCLUZIE - GHID ULTRA-COMPLET NUMPY\n")
print("=" * 80)

print("""
🎉 FELICITĂRI! AI ACUM CEL MAI COMPLET GHID NUMPY! 🎉

📚 CONȚINUT TOTAL:
═══════════════════════════════════════════════════════════════
✅ 30 CAPITOLE COMPREHENSIVE
✅ 600+ EXEMPLE DE COD FUNCȚIONAL
✅ Broadcasting detaliat și avansat
✅ Memory layout și strides
✅ Structured arrays și record arrays
✅ Universal functions (ufuncs) complet
✅ Polynomial operations
✅ FFT (Fast Fourier Transform)
✅ Masked arrays
✅ Matrix decompositions avansate (SVD, QR, Cholesky)
✅ Numerical integration
✅ Interfacing cu C/Fortran/Cython/Numba
✅ Performance optimization detaliat
✅ Best practices la nivel expert

🎯 NIVELURI ACOPERITE:
═══════════════════════════════════════════════════════════════
Beginner    ██████████ Complet
Intermediate██████████ Complet
Advanced    ██████████ Complet
Expert      ██████████ Complet

🚀 NEXT LEVEL SKILLS:
═══════════════════════════════════════════════════════════════
1. Practică zilnic cu probleme reale
2. Contribuie la proiecte open-source
3. Implementează algoritmi din papers
4. Participă la competiții (Kaggle, CodeForces)
5. Explorează biblioteci complementare:
   - SciPy: Scientific computing
   - Pandas: Data manipulation
   - Scikit-learn: Machine learning
   - TensorFlow/PyTorch: Deep learning
   - CuPy: GPU-accelerated NumPy
   - JAX: Autograd și XLA compilation

💡 BIBLIOTECI AVANSATE PENTRU PERFORMANȚĂ:
═══════════════════════════════════════════════════════════════
⚡ Numba - JIT compilation (simplu, rapid)
⚡ Cython - Python → C compilation
⚡ CuPy - NumPy pe GPU (NVIDIA)
⚡ JAX - Autograd + GPU/TPU
⚡ Dask - Parallel NumPy pentru big data
⚡ Xarray - Labeled multi-dimensional arrays

📖 RESURSE CONTINUE:
═══════════════════════════════════════════════════════════════
📚 Documentație:
   - numpy.org/doc/stable
   - NumPy User Guide
   - NumPy Reference

📚 Cărți:
   - "Guide to NumPy" - Travis Oliphant
   - "Elegant SciPy" - Juan Nunez-Iglesias
   - "High Performance Python" - Micha Gorelick

📚 Cursuri:
   - NumPy Tutorials (oficial)
   - DataCamp NumPy Track
   - Real Python NumPy Tutorials
   - YouTube: Corey Schafer, sentdex

🌟 COMUNITATE:
═══════════════════════════════════════════════════════════════
💬 Stack Overflow - numpy tag
💬 NumPy Discussions - github.com/numpy/numpy/discussions
💬 Reddit: r/numpy, r/learnpython
💬 Discord: Python Discord, Scientific Python

🏆 RECUNOAȘTERE EXPERTIZA:
═══════════════════════════════════════════════════════════════
Cu acest ghid, ai acum:
✓ Înțelegere profundă a NumPy
✓ Capacitate de optimizare avansată
✓ Abilitate de debugging complex
✓ Cunoștințe de interfacing C/Fortran
✓ Experiență cu toate structurile de date
✓ Master în broadcasting și vectorizare
✓ Expert în algebră liniară computațională

🎓 EȘTI PREGĂTIT PENTRU:
═══════════════════════════════════════════════════════════════
✅ Data Science roles (mid to senior)
✅ Scientific computing positions
✅ High-performance computing
✅ Quant finance roles
✅ Computer vision engineering
✅ Signal processing
✅ Physics simulations
✅ Bioinformatics
✅ Deep learning engineering

💪 MINDSET PENTRU SUCCES:
═══════════════════════════════════════════════════════════════
1. Gândește vectorizat, nu în loop-uri
2. Măsoară întotdeauna performanța
3. Înțelege memory layout pentru optimizare
4. Broadcasting e prietenul tău
5. Citește cod NumPy din biblioteci populare
6. Contribuie înapoi la comunitate
7. Practică, practică, practică!

🌈 URMĂTORII PAȘI CONCRE

TI:
═══════════════════════════════════════════════════════════════
Săptămâna 1: Implementează algoritmi clasici (sorting, searching)
Săptămâna 2: Rezolvă probleme pe Project Euler cu NumPy
Săptămâna 3: Contribuie la un proiect open-source
Săptămâna 4: Construiește un mini-framework ML cu NumPy pur
Luna 2: Explorează SciPy și integrări avansate
Luna 3: Optimizează cod existent (10x-100x speedup)
Luna 4: Învață CUDA/CuPy pentru GPU computing

════════════════════════════════════════════════════════════════
              MULT SUCCES ÎN CĂLĂTORIA TA NUMPY!
          ACUM EȘTI ECHIPAT PENTRU ORICE PROVOCARE! 🚀
════════════════════════════════════════════════════════════════

📊 STATISTICI FINALE:
══════════════════
Capitole: 30
Exemple: 600+
Linii de cod: 3000+
Concepte: 200+
Timp învățare: 40+ ore
Nivel atins: EXPERT 🏆

════════════════════════════════════════════════════════════════
Data: 2025
Versiune: Ultra-Completă & Optimizată
Autor: Ghid NumPy Master Edition
Status: Production Ready ✅
════════════════════════════════════════════════════════════════
""")

print("=" * 80)
print("FINAL - GHID ULTRA-COMPLET NUMPY - MASTER EDITION")
print("=" * 80)