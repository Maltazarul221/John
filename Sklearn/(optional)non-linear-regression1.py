import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Date ---
# X = [house_size_ft2, crime_rate] (simplificat pentru vizualizare)
X = np.array([
    [800, 0.03],
    [1000, 0.02],
    [1200, 0.04],
    [1500, 0.01],
    [1800, 0.05],
    [2000, 0.03],
    [2200, 0.02],
    [2900, 0.01]
])
y = np.array([150, 180, 200, 250, 300, 330, 360, 400])

# --- 2. Transformarea caracteristicilor în polinoame (grad 2) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Acum X_poly include: [size, crime_rate, size^2, size*crime_rate, crime_rate^2]

# --- 3. Model ---
model = LinearRegression()
model.fit(X_poly, y)

# --- 4. Predicții pentru noi case ---
new_houses = np.array([
    [1600, 0.02],
    [3200, 0.01]
])
new_houses_poly = poly.transform(new_houses)
predictions = model.predict(new_houses_poly)
for house, price in zip(new_houses, predictions):
    print(f"Predicted price for house {house} : ${price*1000:.0f}")

# --- 5. Evaluare ---
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# --- 6. Vizualizare curba preț-dimensiune (simplificat, doar prima caracteristică) ---
plt.scatter(X[:,0], y, color='blue', label='Actual prices')
plt.scatter(new_houses[:,0], predictions, color='green', label='Predicted prices')
x_range = np.linspace(800, 3200, 100).reshape(-1,1)
x_range_poly = poly.transform(np.hstack([x_range, np.full_like(x_range, 0.03)]))  # păstrăm crime_rate fix
plt.plot(x_range, model.predict(x_range_poly), color='red', label='Polynomial fit')
plt.xlabel('House size (ft^2)')
plt.ylabel('Price ($1000)')
plt.legend()
plt.show()
