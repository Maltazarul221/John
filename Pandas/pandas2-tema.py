import pandas as pd

# ==============================================
# Exercițiul 1: Citire pe chunk-uri și afișarea primelor 2 rânduri
# Cerință: Citește CSV-ul pe chunk-uri de 3 rânduri și afișează primele 2 rânduri din fiecare chunk
# ==============================================
chunk_size = 3
for i, chunk in enumerate((pd.read_csv("produse.csv", chunksize=chunk_size))):
    print()


# ==============================================
# Exercițiul 2: Calcul preț mediu pe chunk și global
# Cerință: Calculează prețul mediu pe fiecare chunk și apoi prețul mediu global fără a încărca tot CSV-ul simultan
# ==============================================
for chunk in pd.read_csv("produse.csv", chunksize=3):
    print()


# ==============================================
# Exercițiul 3: Filtrare produse cu stoc < 20 pe chunk
# Cerință: Afișează produsele care au stoc mai mic decât 20 pentru fiecare chunk
# ==============================================
for i, chunk in enumerate(pd.read_csv("produse.csv", chunksize=3)):
    print()


# ==============================================
# Exercițiul 4: Numărarea produselor pe categorie folosind chunk-uri
# Cerință: Calculează numărul de produse per categorie fără a încărca tot CSV-ul simultan
# ==============================================
for chunk in pd.read_csv("produse.csv", chunksize=3):
    print()


# ==============================================
# Exercițiul 5: Crearea unei coloane noi 'venit_lunar' pe chunk
# Cerință: Creează o coloană nouă 'venit_lunar' = preț_unitar * vânzări_lunare pentru fiecare chunk și afișează
# ==============================================
for chunk in pd.read_csv("produse.csv", chunksize=3):
    print()
