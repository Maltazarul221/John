import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Încărcare date reale ---
california = fetch_california_housing()
X = california.data        # toate caracteristicile numerice
y = california.target      # prețurile locuințelor

# --- 2. Vizualizare rapidă a datelor ---
df = pd.DataFrame(X, columns=california.feature_names)
df['Price'] = y
print("5 randuri:")
print(df.sample(5))
