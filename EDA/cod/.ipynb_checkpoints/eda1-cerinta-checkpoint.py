# =======================================================
# Exploratory Data Analysis & Data Preprocessing Exercises
# =======================================================

# =======================================================
# Importăm bibliotecile necesare
# =======================================================
import pandas as pd
import numpy as np

# =======================================================
# Date inițiale
# =======================================================
# Creăm un DataFrame Pandas cu informații despre angajați
data = {
    "id": range(1, 11),
    "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
    "salariu": [4500., 7200., 6800., 5200., 4700., 6000., 5100., 7500., 6400., 5600.],
    "varsta": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
    "experienta_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
}

# Transformăm dicționarul într-un DataFrame
df = pd.DataFrame(data)

# Afișăm datele pentru a verifica structura
print("=== Datele inițiale ===")
print(df)

# =======================================================
# 1️⃣ Handling Missing Data (valori lipsă)
# =======================================================
# CERINȚĂ:
# Simulează valori lipsă pentru 'salariu' și 'vârstă' și înlocuiește-le cu mediana coloanei.

df.loc[1, "salariu"] = np.nan
df.loc[4, "varsta"] = np.nan
print(df.isna().sum())
df["salariu"] = df["salariu"].fillna(df["salariu"].median())        # pune mediana in locul valorilor lipsa
df["varsta"] = df["varsta"].fillna(df["varsta"].median())
print(df)
print(df.isna().sum())

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru valori lipsa

# =======================================================
# 2️⃣ Normalization (Min-Max Scaling)
# =======================================================
# CERINȚĂ:
# Normalizează coloana 'salariu' astfel încât valorile să fie între 0 și 1.
# Formula normalizării: (x - min) / (max - min)

df["salariu_normalizat"] = (df["salariu"] - df["salariu"].min()) / (df["salariu"].max() - df["salariu"].min())
print(df["salariu_normalizat"])

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru normalizare (ex: grafic cu salariu/salariu normalizat)

# =======================================================
# 3️⃣ Standardization / Scalare (Z-score)
# =======================================================
# CERINȚĂ:
# Standardizează coloana 'experiență_ani' astfel încât să aibă media 0 și deviația standard 1.
# Formula scalarii: (x - media) / deviația_standard

df["experienta_ani_std"] = (df["experienta_ani"] - df["experienta_ani"].mean()) / df["experienta_ani"].std()        # std = fct de standardizare
print(df[["experienta_ani", "experienta_ani_std"]])
print(df["experienta_ani"].mean())

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru standardizare (ex: grafic cu salariu/salariu normalizat)

# =======================================================
# 4️⃣ Discretization (Binning)
# =======================================================
# CERINȚĂ:
# Împarte coloana 'vârstă' în 3 categorii: 'Tânăr', 'Matur', 'Senior'

labels = ["Tanar", "Matur", "Senior"]
bins = [0, 30, 40, 100]

df["categorie_varsta"] = pd.cut(df["varsta"], bins = bins, labels = labels)     # exista mai multe implementari, asta este una dintre ele
print(df[["varsta", "categorie_varsta"]])
print(df["categorie_varsta"].value_counts())        # se vede cate persoane sunt in  fiecare categorie

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru normalizare (ex: grafic cu salariu/salariu normalizat)
# ❗ TEMA: Gaseste alta modalitate pentru discretizare

# =======================================================
# 5️⃣ Categorical Imputation (completare categorii lipsă)
# =======================================================
# CERINȚĂ:
# Simulează o valoare lipsă în coloana 'departament' și completează cu cel mai frecvent departament

df.loc[2, "departament"] = np.nan
cmcdp = df["departament"].mode()[0]         # functia mode se aplica pe un array - returneaza un array cu cele mai frecvente elem in ordine
df["departament"] = df["departament"].fillna(cmcdp)
print(df["departament"])
#print(df)

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru categ imputation

# =======================================================
# 6️⃣ Noise Removal (Outlier detection)
# =======================================================
# CERINȚĂ:
# Identifică eventualii outlier-i în coloana 'salariu' folosind deviația standard.
# Detectarea se face comparând fiecare valoare cu media ± factor * deviația standard.

media_salarii = df["salariu"].mean()
deviatia_std = df["salariu"].std()
factor = 1          # distanta fata de medie in variabila
limita_superioara = media_salarii + factor * deviatia_std   # fara paranteze
limita_inferioara = media_salarii - factor * deviatia_std
masca = (df["salariu"] >= limita_inferioara) & (df["salariu"] <= limita_superioara)
# 0 < x < 10
# limita_inferioara <= "salariu" <= limita_superioara
print(masca)
df_clean = df[masca]
print(df_clean)
print(df)

# =======================================================
# 7️⃣ Limiting Outliers (Clipping)
# =======================================================
# CERINȚĂ:
# În loc să eliminăm outlier-ii, putem să îi limităm la anumite valori maxime/minime (percentile).

# =======================================================
# 8️⃣ Grouping și Aggregation
# =======================================================
# CERINȚĂ:
# Grupăm datele după 'departament' și calculăm statistici agregate.

# =======================================================
# 9️⃣ Feature Extraction simplu
# =======================================================
# CERINȚĂ:
# Creăm o nouă caracteristică: raportul salariu/experiență

# =======================================================
# 1️⃣0️⃣ DataFrame final
# =======================================================
# Afișăm DataFrame-ul complet după toate transformările și exercițiile de preprocessing