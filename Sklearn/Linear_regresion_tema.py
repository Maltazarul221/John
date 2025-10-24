# ==========================
# Analiza datelor despre diabet
# ==========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Încărcare date ---
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# --- 2. Creare DataFrame ---
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['Progression'] = y

print("Primele 5 rânduri din setul de date:")
print(df.head(), "\n")
print("Informații despre setul de date:")
print(df.info(), "\n")

# --- 3. Statistici descriptive ---
print("Statistici descriptive:")
print(df.describe(), "\n")

# --- 4. Distribuția valorii țintă ---
plt.figure(figsize=(7,4))
sns.histplot(df["Progression"], bins=25, kde=True)
plt.title("Distribuția progresiei bolii (valoarea țintă)")
plt.xlabel("Progresia bolii")
plt.ylabel("Frecvență")
plt.show()

# --- 5. Matrice de corelație ---
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matricea de corelație între variabile")
plt.show()

# --- 6. Relația dintre IMC (bmi) și progresia bolii ---
plt.figure(figsize=(6,4))
sns.scatterplot(x="bmi", y="Progression", data=df)
plt.title("Relația dintre IMC și progresia bolii")
plt.xlabel("Body Mass Index (bmi)")
plt.ylabel("Progresia bolii")
plt.show()

# --- 7. Relația dintre glicemie (s5) și progresie ---
plt.figure(figsize=(6,4))
sns.scatterplot(x="s5", y="Progression", data=df)
plt.title("Relația dintre glicemie (s5) și progresia bolii")
plt.xlabel("S5 (un marker al glicemiei)")
plt.ylabel("Progresia bolii")
plt.show()

# --- 8. Împărțire train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 9. Antrenare model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 10. Predicții ---
y_pred = model.predict(X_test)

# --- 11. Evaluare model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 Coeficientul de determinare (R²): {r2:.2f}\n")

# --- 12. Grafic Predicții vs Valori reale ---
plt.figure(figsize=(6,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valori reale")
plt.ylabel("Predicții")
plt.title("Predicții vs Valori reale (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()

# --- 13. Importanța caracteristicilor ---
coef_df = pd.DataFrame({
    "Caracteristică": diabetes.feature_names,
    "Coeficient": model.coef_
}).sort_values(by="Coeficient", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Coeficient", y="Caracteristică", data=coef_df, palette="viridis")
plt.title("Importanța caracteristicilor în modelul de regresie liniară")
plt.show()

print("Coeficienții modelului:\n", coef_df)
