# --- Import librării ---
import numpy as np  # Pentru operații numerice și generare de array-uri
import pandas as pd  # Pentru manipularea dataset-urilor sub formă de DataFrame/Series
import matplotlib.pyplot as plt  # Pentru vizualizări grafice

# --- Librării sklearn ---
from sklearn.datasets import load_iris  # Pentru a încărca un dataset clasic de clasificare
from sklearn.model_selection import train_test_split, StratifiedKFold
# train_test_split: împarte datele în set de antrenament și test
# StratifiedKFold: pentru cross-validation păstrând proporția claselor

from sklearn.linear_model import LogisticRegression  # Model de clasificare
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
# Metrici pentru evaluarea performanței modelului

# ============================================================
# --- Încarcă dataset ---
# ============================================================

# load_iris() returnează 2 obiecte:
# X = valorile numerice (features) ale fiecărei flori (4 dimensiuni: SepalLength, SepalWidth, PetalLength, PetalWidth)
# y = clasele (target), adică specia plantei: 0 = setosa, 1 = versicolor, 2 = virginica
X, y = load_iris(return_X_y=True)

# Transformăm X în DataFrame pentru manipulare ușoară și adăugăm nume de coloane
X = pd.DataFrame(X, columns=["SepalLength","SepalWidth","PetalLength","PetalWidth"])

# Transformăm y într-un Series cu nume
y = pd.Series(y, name="Species")

# ============================================================
# --- Vizualizare date inițiale ---
# ============================================================

# Arătăm câteva rânduri ale datasetului pentru a vedea valorile numerice
print("Sample features X:")
print(X.sample(5))  # Afișează 5 rânduri aleatorii

# Verificăm distribuția claselor pentru a înțelege dacă datele sunt echilibrate
print("\nDistribuția claselor y:")
print(y.value_counts())  # Afișează câte exemple sunt din fiecare clasă
