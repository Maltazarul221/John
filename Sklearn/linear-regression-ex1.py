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

# --- 4. Crearea și antrenarea modelului ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. Predicții pentru setul de test ---
y_pred = model.predict(X_test)

# --- 6. Evaluarea performanței modelului ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valori reale')
plt.ylabel('Predicții')
plt.title('Predicții vs Valori reale')
plt.text(0.05, 0.9, f'R^2 = {r2:.2f}', transform=plt.gca().transAxes)
plt.show()

# # --- 7. Vizualizare completă cu subploturi ---
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#
# # 7a. Predicții vs valori reale
# axs[0,0].scatter(y_test, y_pred, color='blue', alpha=0.7)
# axs[0,0].plot([y_test.min(), y_test.max()],
#               [y_test.min(), y_test.max()], 'r--', lw=2)
# axs[0,0].set_title('Predicții vs Valori reale')
# axs[0,0].set_xlabel('Valori reale')
# axs[0,0].set_ylabel('Predicții')
# axs[0,0].text(0.05, 0.9, f'R^2 = {r2:.2f}', transform=axs[0,0].transAxes)
#
# # 7b. Reziduuri (erori)
# residuals = y_test - y_pred
# axs[0,1].scatter(y_test, residuals, color='green', alpha=0.7)
# axs[0,1].axhline(0, color='r', linestyle='--', lw=2)
# axs[0,1].set_title('Reziduuri (erori)')
# axs[0,1].set_xlabel('Valori reale')
# axs[0,1].set_ylabel('Reziduu')
# axs[0,1].text(0.05, 0.9, f'MSE = {mse:.2f}', transform=axs[0,1].transAxes)
#
# # 7c. Date train vs test
# axs[1,0].scatter(range(len(y_train)), y_train, color='blue', label='Train', alpha=0.7)
# axs[1,0].scatter(range(len(y_train), len(y_train)+len(y_test)), y_test,
#                   color='orange', label='Test', alpha=0.7)
# axs[1,0].set_title('Date train vs test')
# axs[1,0].set_xlabel('Index')
# axs[1,0].set_ylabel('Progresie diabet')
# axs[1,0].legend()
#
# # 7d. Distribuția predicțiilor
# axs[1,1].hist(y_pred, bins=20, color='purple', alpha=0.7)
# axs[1,1].set_title('Distribuția predicțiilor modelului')
# axs[1,1].set_xlabel('Predicție')
# axs[1,1].set_ylabel('Frecvență')
#
# plt.tight_layout()
# plt.show()
