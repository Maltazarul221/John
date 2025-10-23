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

# ===========================
# 1 figură cu toate graficele relevante
# ===========================
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

# 1. Violin + swarm plot
sns.violinplot(x="Categorie", y="Valoare1", data=data, inner=None, color="lightblue", ax=axes[0])
sns.swarmplot(x="Categorie", y="Valoare1", data=data, color="k", alpha=0.6, ax=axes[0])
axes[0].set_title("Violin + Swarm Valoare1 pe Categorie")

# 2. Histogram pe Subcategorie (manual, în loc de FacetGrid)
sns.histplot(data=data[data['Subcategorie']=="X"], x="Valoare1", hue="Categorie", bins=15, kde=True, ax=axes[1])
axes[1].set_title("Histogram Subcategorie X")
sns.histplot(data=data[data['Subcategorie']=="Y"], x="Valoare1", hue="Categorie", bins=15, kde=True, ax=axes[2])
axes[2].set_title("Histogram Subcategorie Y")

# 3. Scatterplot Valoare1 vs Valoare2
sns.scatterplot(x="Valoare1", y="Valoare2", hue="Categorie", data=data, ax=axes[3])
axes[3].set_title("Scatter Valoare1 vs Valoare2")

# 4. Heatmap corelație Valoare1/Valoare2 per categorie
for i, cat in enumerate(data['Categorie'].unique()):
    subset = data[data['Categorie']==cat][['Valoare1','Valoare2']]
    corr = subset.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=False, ax=axes[4+i])
    axes[4+i].set_title(f"Heatmap Categorie {cat}")

# 5. Boxplot pe Subcategorie cu hue
sns.boxplot(x="Subcategorie", y="Valoare1", hue="Categorie", data=data, ax=axes[7])
axes[7].set_title("Boxplot Valoare1 pe Subcategorie cu hue")

# 6. KDE 2D
sns.kdeplot(x="Valoare1", y="Valoare2", fill=True, cmap="Blues", data=data, ax=axes[8])
axes[8].set_title("KDE 2D Valoare1 vs Valoare2")

plt.tight_layout()
plt.show()
