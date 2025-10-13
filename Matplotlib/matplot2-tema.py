import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dataset
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

# ==============================================
# Cerințe grafice:
# ==============================================
# 1. Line plot: stoc cumulativ pe produse
# 2. Bar plot: preț mediu pe categorie
# 3. Horizontal bar plot: stoc mediu pe furnizor
# 4. Scatter plot: vânzări lunare vs preț unitare
# 5. Histogram: distribuția prețurilor unitare
# 6. Pie chart: proporția produselor pe categorii
# 7. Area plot: stoc cumulativ pe produse
# 8. Step plot: stoc cumulativ pe produse
# 9. Subplots: scatter vânzări lunare vs preț unitar și stoc vs preț unitar

# ================================
# 1. Line plot – stoc cumulativ pe produse
# ================================
plt.figure(figsize=(8, 5))
plt.plot(df["produs"], df["stoc"].cumsum(), marker='o')
plt.title("Stoc cumulativ pe produse")
plt.xlabel("Produs")
plt.ylabel("Stoc cumulativ")
plt.grid(True)
plt.show()

# ================================
# 2. Bar plot – preț mediu pe categorie
# ================================
plt.figure(figsize=(8, 5))
df.groupby("categorie")["preț_unitar"].mean().plot(kind="bar")
plt.title("Preț mediu pe categorie")
plt.ylabel("Preț mediu (lei)")
plt.show()

# ================================
# 3. Horizontal bar plot – stoc mediu pe furnizor
# ================================
plt.figure(figsize=(8, 5))
df.groupby("furnizor")["stoc"].mean().plot(kind="barh", color='orange')
plt.title("Stoc mediu pe furnizor")
plt.xlabel("Stoc mediu")
plt.show()

# ================================
# 4. Scatter plot – vânzări lunare vs preț unitar
# ================================
plt.figure(figsize=(8, 5))
plt.scatter(df["preț_unitar"], df["vânzări_lunare"], color='green')
plt.title("Vânzări lunare vs Preț unitar")
plt.xlabel("Preț unitar (lei)")
plt.ylabel("Vânzări lunare")
plt.grid(True)
plt.show()

# ================================
# 5. Histogram – distribuția prețurilor unitare
# ================================
plt.figure(figsize=(8, 5))
plt.hist(df["preț_unitar"], bins=8, color='skyblue', edgecolor='black')
plt.title("Distribuția prețurilor unitare")
plt.xlabel("Preț unitar (lei)")
plt.ylabel("Frecvență")
plt.show()

# ================================
# 6. Pie chart – proporția produselor pe categorii
# ================================
plt.figure(figsize=(6, 6))
df["categorie"].value_counts().plot(kind="pie", autopct="%.1f%%")
plt.title("Proporția produselor pe categorii")
plt.ylabel("")
plt.show()

# ================================
# 7. Area plot – stoc cumulativ pe produse
# ================================
plt.figure(figsize=(8, 5))
plt.fill_between(df["produs"], df["stoc"].cumsum(), color='lightgreen', alpha=0.6)
plt.title("Stoc cumulativ (area plot)")
plt.xlabel("Produs")
plt.ylabel("Stoc cumulativ")
plt.show()

# ================================
# 8. Step plot – stoc cumulativ pe produse
# ================================
plt.figure(figsize=(8, 5))
plt.step(df["produs"], df["stoc"].cumsum(), where='mid', color='red')
plt.title("Stoc cumulativ (step plot)")
plt.xlabel("Produs")
plt.ylabel("Stoc cumulativ")
plt.grid(True)
plt.show()

# ================================
# 9. Subplots – două scatter plots
# ================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df["preț_unitar"], df["vânzări_lunare"], color='purple')
axes[0].set_title("Vânzări lunare vs Preț unitar")
axes[0].set_xlabel("Preț unitar (lei)")
axes[0].set_ylabel("Vânzări lunare")

axes[1].scatter(df["preț_unitar"], df["stoc"], color='teal')
axes[1].set_title("Stoc vs Preț unitar")
axes[1].set_xlabel("Preț unitar (lei)")
axes[1].set_ylabel("Stoc")

plt.tight_layout()
plt.show()
