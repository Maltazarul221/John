import pandas as pd
from tqdm import tqdm

# ==============================================
# Exercițiul 1: Citire pe chunk-uri și afișarea primelor 2 rânduri
# Cerință: Citește CSV-ul pe chunk-uri de 3 rânduri și afișează primele 2 rânduri din fiecare chunk
# ==============================================
chunk_size = 3
for i, chunk in enumerate(tqdm(pd.read_csv("angajati.csv", chunksize=chunk_size), desc="Processing chunks")):
    print(f"--- Chunk {i+1} ---")
    print(chunk.head(2))


# ==============================================
# Exercițiul 2: Calcul medie salariu pe chunk și global
# Cerință: Calculează salariul mediu pe fiecare chunk și apoi salariul mediu global fără a încărca tot CSV-ul simultan
# ==============================================
for chunk in pd.read_csv("angajati.csv", chunksize=3):
    print()
print("Salariu mediu global:")


# ==============================================
# Exercițiul 3: Filtrare angajați cu salariu > 6000 pe chunk
# Cerință: Afișează angajații cu salariu > 6000 pentru fiecare chunk
# ==============================================
for i, chunk in enumerate(pd.read_csv("angajati.csv", chunksize=3)):
    print()

# ==============================================
# Exercițiul 4: Numărarea angajaților pe departament folosind chunk-uri
# Cerință: Calculează numărul de angajați per departament fără a încărca tot CSV-ul simultan
# ==============================================
for chunk in pd.read_csv("angajati.csv", chunksize=3):
    print()

# ==============================================
# Exercițiul 5: Crearea unei coloane noi 'salariu_anual' pe chunk
# Cerință: Creează o coloană nouă 'salariu_anual' = salariu * 12 pentru fiecare chunk și afișează
# ==============================================
for i, chunk in enumerate(pd.read_csv("angajati.csv", chunksize=3)):
    print()
