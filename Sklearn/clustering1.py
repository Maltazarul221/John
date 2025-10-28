# clustering_algorithms_demo_with_context.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -------------------------------------------------------------
# 1. Generarea datelor de intrare (scenariu realist)
# -------------------------------------------------------------
# Imaginăm că datele reprezintă 500 de clienți ai unui supermarket.
# Fiecare client este caracterizat prin două trăsături:
#   - Venitul lunar (în mii de lei)
#   - Frecvența medie a vizitelor la magazin (număr pe lună)
#
# Vom genera artificial aceste date astfel încât să existe 4 grupuri
# distincte de clienți (ex: familii, tineri singuri, pensionari, etc.)
X, y_true = make_blobs(
    n_samples=500,      # număr total de clienți
    centers=4,          # presupunem că există 4 tipuri principale de clienți
    cluster_std=0.60,   # cât de răspândiți sunt clienții din fiecare grup
    random_state=42
)

# -------------------------------------------------------------
# 2. Aplicăm mai multe algoritme de clustering
# -------------------------------------------------------------
# Fiecare algoritm va încerca să descopere, fără etichete predefinite,
# cum se grupează acești clienți în funcție de venitul și frecvența lor.

models = {
    "K-Means": KMeans(n_clusters=4, random_state=42),
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.9, min_samples=5),
    "Gaussian Mixture Model": GaussianMixture(n_components=4, random_state=42)
}

results = {}

for name, model in models.items():
    # Majoritatea modelelor folosesc metoda fit_predict pentru a genera etichete de cluster
    labels = model.fit_predict(X) if name != "Gaussian Mixture Model" else model.fit_predict(X)

    # Pentru DBSCAN, unele puncte pot fi etichetate ca zgomot (-1)
    valid = labels != -1
    if np.sum(valid) > 1:
        silhouette = silhouette_score(X[valid], labels[valid])
        dbi = davies_bouldin_score(X[valid], labels[valid])
    else:
        silhouette, dbi = np.nan, np.nan

    # Pentru K-Means putem calcula SSE (Sum of Squared Errors)
    sse = model.inertia_ if name == "K-Means" else np.nan

    results[name] = {
        "Silhouette": silhouette,
        "DBI": dbi,
        "SSE": sse
    }

    # ---------------------------------------------------------
    # 3. Vizualizare
    # ---------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis')
    plt.title(f"{name} Clustering")
    plt.xlabel("Venit lunar (mii lei)")
    plt.ylabel("Frecvență vizite/lună")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# 4. Evaluarea performanței clusteringului
# -------------------------------------------------------------
print("\n=== Evaluare Clustering ===")
for name, metrics in results.items():
    print(f"\n{name}")
    print(f"  Silhouette Score: {metrics['Silhouette']:.3f}")
    print(f"  Davies-Bouldin Index: {metrics['DBI']:.3f}")
    if not np.isnan(metrics['SSE']):
        print(f"  SSE: {metrics['SSE']:.3f}")

# -------------------------------------------------------------
# 5. Determinarea numărului optim de clustere (Elbow Method)
# -------------------------------------------------------------
# În scenariul nostru, vrem să aflăm câte tipuri distincte de clienți există.
# Vom aplica metoda "cotului" (Elbow Method) pentru a analiza cum se schimbă
# eroarea SSE în funcție de numărul de clustere ales.

sse_list = []
k_values = range(2, 10)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    sse_list.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_values, sse_list, marker='o')
plt.title("Metoda Cotului - Determinarea numărului optim de clustere")
plt.xlabel("Numărul de clustere (k)")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.tight_layout()
plt.show()
