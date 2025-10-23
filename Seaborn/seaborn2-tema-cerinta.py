# =======================================================
# Vizualizări cu Seaborn pentru date produse
# =======================================================

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='categorie', y='preț_unitar', palette='Set2')
plt.title('Distribuția Prețurilor Unitare pe Categorie', fontsize=14, fontweight='bold')
plt.xlabel('Categorie', fontsize=12)
plt.ylabel('Preț Unitar (RON)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. VIOLIN PLOT – Distribuția stocurilor pe categorie

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='categorie', y='stoc', palette='muted')
plt.title('Distribuția Stocurilor pe Categorie', fontsize=14, fontweight='bold')
plt.xlabel('Categorie', fontsize=12)
plt.ylabel('Stoc (unități)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. HEATMAP – Corelația între preț, stoc și vânzări lunare

corr_data = df[['preț_unitar', 'stoc', 'vânzări_lunare']].corr()
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Corelația între Preț, Stoc și Vânzări Lunare', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 4. KDE PLOT – Densitatea prețurilor unitare

