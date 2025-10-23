import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Date multiple caracteristici (multivariate) ---
# Coloane: [house_size_ft2, crime_rate, num_rooms]
X = np.array([
    [800, 0.03, 2],
    [1000, 0.02, 3],
    [1200, 0.04, 3],
    [1500, 0.01, 4],
    [1800, 0.05, 4],
    [2000, 0.03, 5],
    [2200, 0.02, 5],
    [2500, 0.01, 6]
])
y = np.array([150, 180, 200, 250, 300, 330, 360, 400])  # Prețurile

# --- 2. Definirea și antrenarea modelului ---
model = LinearRegression()
model.fit(X, y)

# --- 3. Coeficienți și intercept ---
print("Intercept (θ0):", model.intercept_)
print("Coefficients (θ1, θ2, θ3):", model.coef_)

# --- 4. Predicții pentru noi case ---
new_houses = np.array([
    [1600, 0.02, 4],
    [3200, 0.01, 6]
])
predictions = model.predict(new_houses)
for house, price in zip(new_houses, predictions):
    print(f"Predicted price for house {house} : ${price*1000:.0f}")

# --- 5. Evaluarea modelului ---
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
