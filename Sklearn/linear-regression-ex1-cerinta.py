import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Încărcare date reale ---
diabetes = load_diabetes()
X = diabetes.data       # toate caracteristicile numerice
y = diabetes.target     # progresia bolii

# --- 2. Vizualizare rapidă a datelor ---
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['Progression'] = y
print("5 randuri:")
print(df.sample(5))

# --- 3. Împărțire train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
