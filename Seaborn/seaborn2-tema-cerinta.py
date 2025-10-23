# =======================================================
# Vizualizări cu Seaborn pentru date produse
# =======================================================

import pandas as pd

# =======================================================
# Date produse
# =======================================================
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
print(df)

# =======================================================
# Cerințe / Exercitii
# =======================================================
# 1. BOX PLOT – Distribuția prețurilor unitare pe categorie
# 2. VIOLIN PLOT – Distribuția stocurilor pe categorie
# 3. HEATMAP – Corelația între preț, stoc și vânzări lunare
# 4. KDE PLOT – Densitatea prețurilor unitare
