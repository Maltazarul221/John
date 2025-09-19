import numpy as np

# 1. Încarcă Date Financiare în Array-uri NumPy
# Cerință: Încarcă fișierul 'financial_data.csv' într-un array structurat NumPy,
# incluzând coloanele: Company (string), Quarter (string), Revenue (float), Profit (float), Expenses (float).
# Inspectează primele câteva rânduri pentru a verifica încărcarea corectă.

dtype=[
    ('Company', 'U20'), #unicode de 20 char
    ('Quarter', 'U4'),  # unicode de 4 char
    ('Revenue', 'f8'),  # float pe 8 bytes
    ('Profit', 'f8'),  # float pe 8 bytes
    ('Expenses', 'f8')  # float pe 8 bytes
]

data = np.genfromtxt('financial_data.csv', delimiter = ',', skip_header=1, dtype=dtype)
print("#1 Încărcare date financiare în array NumPy:\n", data, "\n")

# 2. Calculează Marja de Profit pentru fiecare companie și trimestru
# Cerință: Calculează Marja de Profit = Profit / Revenue pentru fiecare rând.
# Asigură-te că împarți la zero în mod corect pentru a evita erorile.
print("#2 Marja de Profit per companie și trimestru:\n", "\n")

# 3. Calculează Rata Cheltuielilor pentru fiecare companie și trimestru
# Cerință: Calculează Rata Cheltuielilor = Expenses / Revenue pentru fiecare rând.
# Aceasta arată ce fracțiune din venituri este cheltuită pe cheltuieli.
print("#3 Rata Cheltuielilor per companie și trimestru:\n", "\n")

# 4. Analizează Creșterea Veniturilor Trimestriale
# Cerință: Pentru fiecare companie, calculează rata de creștere a veniturilor între trimestre consecutive:
# Rata Creșterii = (Venituri_trimestru_următor - Venituri_trimestru_curent) / Venituri_trimestru_curent
print("#4 Creștere Venituri Trimestriale per companie:\n", "\n")

# 5. Identifică Companiile cu Creștere Ridicată
# Cerință: Determină perechi companie-trimestru unde creșterea veniturilor depășește percentila 10
# a tuturor ratelor de creștere trimestrială. Aceasta identifică companii cu tendințe de creștere peste minim.
print("#5 Companii cu Creștere Ridicată (peste percentila 10 a creșterii veniturilor):\n", "\n")

# 6. Numără Trimestrele cu Creștere Ridicată pentru fiecare companie
# Cerință: Pentru fiecare companie, numără trimestrele unde creșterea veniturilor depășește pragul
# percentila 10, pentru a evalua consistența creșterii.
print("#6 Număr Trimestre cu Creștere Ridicată per companie:\n", "\n")

# 7. Evaluează Oportunitățile de Investiții
# Cerință: Clasifică companiile pe baza (1) numărului de trimestre cu creștere ridicată și (2) marja medie de profit.
# Afișează companiile în ordine descrescătoare a potențialului investițional, folosind un scor combinat.
print("#7 Clasament Companii după Potențial Investițional:\n", "\n")
