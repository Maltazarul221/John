import pandas as pd
import numpy as np

# ==============================================
# Generarea datasetului
# ==============================================
data = {
    "id": range(1, 11),
    "nume": ["Ana", "Bogdan", "Carmen", "Dan", "Elena", "Florin", "Gina", "Horațiu", "Ioana", "Jianu"],
    "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
    "oraș": ["București", "Cluj", "Iași", "Timișoara", "Cluj", "București", "Iași", "Cluj", "Timișoara", "București"],
    "salariu": [4500., 7200., 6800., 5200., 4500., 4500., 5100., 7500., 6400., 5600.],
    "vârstă": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
    "experiență_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
}

df = pd.DataFrame(data)
print("=== Dataset inițial ===")
print(df)

# ==============================================
# Exercițiul 1 - Explorarea datelor
# ==============================================
# Cerință: Afișează primele 3 rânduri, informațiile despre DataFrame și statisticile descriptive.
print("\n=== Exercițiul 1 ===")
print("Primele 3 rânduri:")
print(df.head(3))
print("Info DataFrame:")
print(df.info())
print("Statistici descriptive:")
print(df.describe())

# ==============================================
# Exercițiul 2 - Selecția datelor
# ==============================================
# Cerință: Selectează coloana nume, apoi coloanele nume și oraș, apoi toate persoanele cu vârstă > 30.
print("\n=== Exercițiul 2 ===")
print("Coloana nume:")
print(df['nume'])
print("Coloanele nume și oraș:")
print(df[['nume','oraș','departament']])
print("Persoanele cu vârstă > 30:")
print(df[df['vârstă'] > 30])

# ==============================================
# Exercițiul 3 - Indexare și filtrare text
# ==============================================
# Cerință: Selectează rândurile unde oraș conține 'Cluj'.
print("\n=== Exercițiul 3 ===")
print("Rândurile unde oraș conține 'Cluj':")
print(df[df['oraș'].str.contains('Cluj')])

# ==============================================
# Exercițiul 4 - Operări pe coloane
# ==============================================
# Cerință: Creează o coloană nouă cu salariul anual și modifică salariul pentru cei cu experiență > 10 ani (+10%).
print("\n=== Exercițiul 4 ===")
print("Salariu anual și ajustare salariu pentru experiență > 10 ani:")
mask = df['experiență_ani'] > 10
df.loc[mask,'salariu'] = df.loc[mask,'salariu'] * 1.10
df['salariu-anual'] = df['salariu'] * 12
print(df[['nume','salariu','salariu-anual']])

# ==============================================
# Exercițiul 5 - Valori lipsă
# ==============================================
# Cerință: Adaugă câteva valori lipsă și gestionează-le.
print("\n=== Exercițiul 5 ===")
print("Valori lipsă:")
df.loc[2,'oraș'] = np.nan
df.loc[5,'salariu'] = np.nan
print(df.isna().sum())
print(df.isna())
print(df.loc[:,df.isna().any()])
print("După completarea valorilor lipsă:")
df['oraș'] = df['oraș'].fillna('Necunoscut')
df['salariu'] = df['salariu'].fillna(df['salariu'].mean())
print(df[['salariu','oraș']])

# ==============================================
# Exercițiul 6 - Grupare și agregare
# ==============================================
# Cerință: Grupează angajații după departament și calculează salariul mediu și vârsta medie.
print("\n=== Exercițiul 6 ===")
print("Salariu mediu și vârsta medie pe departament:")
print(df.groupby('departament').agg({'salariu':'mean','vârstă':'mean'}))
print(df.groupby('departament').agg({'salariu':'min','vârstă':'max'}))
print(df.groupby(['departament','oraș']).agg({'salariu':'mean','vârstă':'mean'}))

# ==============================================
# Exercițiul 7 - Join/Merge
# ==============================================
# Cerință: Creează un al doilea DataFrame cu bonusuri și alătură-l.
print("\n=== Exercițiul 7 ===")
print("DataFrame combinat cu bonusuri:")
data_2 = {
    "id": [1,3,5,7,9,10,11,48,19,22],
    "bonus": [500., 700., 65., 5000., 3222., 123., 500., 7500., 6400., 5600.]
}
df_2 = pd.DataFrame(data_2)
merge_left = pd.merge(df, df_2, on = 'id', how = 'left')
merge_right = pd.merge(df, df_2, on = 'id', how = 'right')
merge_inner = pd.merge(df, df_2, on = 'id', how = 'inner')

print(merge_left[['id','nume', 'bonus']])
print("\n",merge_right[['id','nume', 'bonus']])
print("\n",merge_inner[['id','nume', 'bonus']])

# ==============================================
# Exercițiul 8 - Pivot și reshaping
# ==============================================
# Cerință: Creează un tabel pivot cu salariul mediu pe oraș și departament.
print("\n=== Exercițiul 8 ===")
print("Tabel pivot cu salariul mediu:")
pivot_table = df.pivot_table(
    values = 'salariu',
    index = 'oraș',
    columns = 'departament',
    aggfunc = 'mean'
)
print(pivot_table)

# ==============================================
# Exercițiul 9 - Sortare
# ==============================================
# Cerință: Sortează după salariu descrescător, apoi după vârstă crescător.
print("\n=== Exercițiul 9 ===")
print("Date sortate după salariu descrescător și vârstă crescător:")
sorted = df.sort_values(by = ['salariu','vârstă'], ascending = [False, True])
print(sorted[['nume','salariu', 'vârstă']])

# ==============================================
# Exercițiul 10 - Export și import
# ==============================================
# Cerință: Salvează datasetul într-un CSV și citește-l înapoi.
print("\n=== Exercițiul 10 ===")
print("Dataset exportat și reîncărcat:")
df.to_csv('./angajati.csv', index = False)
data_3 = pd.read_csv('./angajati.csv')
print(data_3)
