import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dataset fictiv
np.random.seed(42)
data = pd.DataFrame({
    "Categorie": np.random.choice(["A", "B", "C"], 200),
    "Subcategorie": np.random.choice(["X", "Y"], 200),
    "Valoare1": np.random.normal(loc=50, scale=10, size=200),
    "Valoare2": np.random.normal(loc=30, scale=5, size=200)
})

# ==============================================
# 1. Violin + swarm plot combinat (distribuție + puncte)
plt.figure(figsize=(8,5))
sns.violinplot(x="Categorie", y="Valoare1", data=data, inner=None, color="lightblue")
sns.swarmplot(x="Categorie", y="Valoare1", data=data, color="k", alpha=0.6)
plt.title("Distribuție Valoare1 pe Categorie (Violin + Swarm)")
plt.show()

# ==============================================
# 2. FacetGrid: histograma Valoare1 pe Subcategorie
g = sns.FacetGrid(data, col="Subcategorie", hue="Categorie", height=4, aspect=1)
g.map(sns.histplot, "Valoare1", bins=15, kde=True)
g.add_legend()
plt.show()

# ==============================================
# 3. Pairplot cu hue
sns.pairplot(data, vars=["Valoare1","Valoare2"], hue="Categorie", corner=True)
plt.show()

# ==============================================
# 4. Heatmap: corelație între Valoare1 și Valoare2 per Categorie
plt.figure(figsize=(6,4))
for cat in data['Categorie'].unique():
    subset = data[data['Categorie'] == cat][['Valoare1','Valoare2']]
    corr = subset.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=False)
    plt.title(f"Corelație Valoare1 vs Valoare2 - Categorie {cat}")
    plt.show()

# ==============================================
# 5. Catplot: boxplot automat pe Subcategorie cu hue
sns.catplot(x="Subcategorie", y="Valoare1", hue="Categorie", kind="box", data=data, height=5, aspect=1.2)
plt.title("Boxplot Valoare1 pe Subcategorie cu hue")
plt.show()

# ==============================================
# 6. KDE 2D (bivariate) cu contour
plt.figure(figsize=(7,5))
sns.kdeplot(x="Valoare1", y="Valoare2", data=data, fill=True, cmap="Blues")
plt.title("Densitate Valoare1 vs Valoare2")
plt.show()

# ==============================================
# 7. Stripplot + jitter (vizualizare individuala pe categorie)
plt.figure(figsize=(8,5))
sns.stripplot(x="Categorie", y="Valoare2", data=data, hue="Subcategorie", dodge=True, jitter=0.3)
plt.title("Valoare2 individual pe Categorie")
plt.show()
