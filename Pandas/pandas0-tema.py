import pandas as pd
import numpy as np

# ==============================================
# Generarea datasetului
# ==============================================
data = {
    "id": range(1, 11),
    "produs": ["Laptop", "Telefon", "Tabletă", "Monitor", "Mouse", "Tastatură", "Căști", "Router", "SSD", "HDD"],
    "categorie": ["Electronice", "Electronice", "Electronice", "Periferice", "Periferice", "Periferice", "Periferice", "Electronice", "Stocare", "Stocare"],
    "furnizor": ["Furnizor A", "Furnizor B", "Furnizor C", "Furnizor A", "Furnizor B", "Furnizor C", "Furnizor A", "Furnizor B", "Furnizor C", "Furnizor A"],
    "preț_unitar": [3200., 1800., 1500., 900., 120., 250., 180., 700., 450., 300.],
    "stoc": [10, 25, 15, 20, 50, 40, 30, 18, 22, 35],
    "vânzări_lunare": [5, 20, 12, 7, 35, 28, 25, 10, 15, 18]
}

df = pd.DataFrame(data)
print("=== Dataset inițial ===")
print(df)

# ==============================================
# Exercițiul 1 - Explorarea datelor
# ==============================================
# Cerință: Afișează ultimele 5 rânduri, informațiile despre DataFrame și statisticile descriptive.
print("\n=== Exercițiul 1 ===")
print("Ultimele 5 rânduri:")
print()
print("\nInfo DataFrame:")
print()
print("\nStatistici descriptive:")
print()

# ==============================================
# Exercițiul 2 - Selecția datelor
# ==============================================
# Cerință: Selectează coloana categorie, apoi coloanele produs și furnizor, apoi toate produsele cu stoc < 20.
print("\n=== Exercițiul 2 ===")
print("Coloana categorie:")
print()
print("\nColoanele produs și furnizor:")
print()
print("\nProdusele cu stoc < 20:")
print()

# ==============================================
# Exercițiul 3 - Indexare și filtrare text
# ==============================================
# Cerință: Selectează rândurile unde furnizor conține 'B'.
print("\n=== Exercițiul 3 ===")
print("Rândurile unde furnizor conține 'B':")
print()

# ==============================================
# Exercițiul 4 - Operări pe coloane
# ==============================================
# Cerință: Creează o coloană nouă cu valoarea totală stoc*preț și identifică produsele cu vânzări lunare > 15.
print("\n=== Exercițiul 4 ===")
print("Valoarea totală stoc și produse cu vânzări > 15:")
print()

# ==============================================
# Exercițiul 5 - Valori lipsă
# ==============================================
# Cerință: Adaugă câteva valori lipsă în coloana preț și stoc și gestionează-le completând cu mediana.
print("\n=== Exercițiul 5 ===")
print("Date cu valori lipsă:")
print()
print("\nDupă completarea valorilor lipsă:")
print()

# ==============================================
# Exercițiul 6 - Grupare și agregare
# ==============================================
# Cerință: Grupează produsele după furnizor și calculează prețul mediu și stocul mediu.
print("\n=== Exercițiul 6 ===")
print("Preț mediu și stoc mediu pe furnizor:")
print()

# ==============================================
# Exercițiul 7 - Join/Merge
# ==============================================
# Cerință: Creează un al doilea DataFrame cu discount-uri și alătură-l la primul.
print("\n=== Exercițiul 7 ===")
print("DataFrame combinat cu discount:")
print()

# ==============================================
# Exercițiul 8 - Pivot și reshaping
# ==============================================
# Cerință: Creează un tabel pivot cu stocul mediu pe categorie și furnizor.
print("\n=== Exercițiul 8 ===")
print("Tabel pivot cu stocul mediu:")
print()

# ==============================================
# Exercițiul 9 - Sortare
# ==============================================
# Cerință: Sortează după preț crescător, apoi după vânzări descrescător.
print("\n=== Exercițiul 9 ===")
print("Date sortate după preț crescător și vânzări descrescător:")
print()

# ==============================================
# Exercițiul 10 - Export și import
# ==============================================
# Cerință: Salvează datasetul într-un CSV și citește-l înapoi.
print("\n=== Exercițiul 10 ===")
print("Dataset exportat și reîncărcat:")
print()
