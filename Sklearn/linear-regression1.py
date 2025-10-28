# Importăm bibliotecile necesare
import numpy as np  # Pentru manipularea matricelor și vectorilor
from sklearn.linear_model import LinearRegression  # Modelul de regresie liniară
from sklearn.metrics import mean_squared_error      # Pentru evaluarea performanței modelului
import matplotlib.pyplot as plt                     # Pentru vizualizarea datelor și a regresiei

# --- 1. Definirea dataset-ului simplu ---
# X reprezintă caracteristica independentă (house size in square feet)
# y reprezintă variabila dependentă (house price în $1000)
# reshape(-1, 1) este necesar deoarece sklearn așteaptă un array 2D pentru X
X = np.array([
    1020, 870, 1455, 1190, 830, 1680, 1380, 950, 1825, 1240,
    1105, 1600, 940, 2005, 1780, 1335, 820, 1520, 1765, 1075,
    1185, 1840, 1395, 1570, 900, 1210, 1710, 990, 2000, 1265,
    850, 1625, 1150, 1735, 1370, 1085, 1650, 940, 1820, 1305,
    870, 1540, 1200, 1750, 1400, 1015, 1675, 950, 1890, 1325
]).reshape(-1, 1)

y = np.array([
    180, 155, 260, 210, 160, 300, 240, 170, 320, 225,
    200, 290, 175, 355, 310, 235, 150, 275, 330, 205,
    215, 340, 245, 285, 165, 220, 315, 190, 360, 230,
    155, 295, 205, 325, 250, 200, 310, 180, 345, 240,
    160, 280, 215, 335, 255, 185, 305, 170, 350, 245
])

# Explicație:
# - Fiecare "sample" este un apartament/house
# - X = mărimea casei, y = prețul acesteia
# - Avem un exemplu de regresie liniară univariată

# --- 2. Crearea modelului de regresie liniară ---
# LinearRegression() creează un obiect model care va învăța coeficienții θ0 și θ1
model = LinearRegression()

# --- 3. Antrenarea modelului ---
# Fit antrenează modelul, adică găsește cei mai buni coeficienți care minimizează eroarea pătratică
model.fit(X, y)

# --- 4. Inspectarea parametrilor modelului ---
# model.intercept_ = θ0 (interceptul liniei, unde aceasta intersectează axa y)
# model.coef_ = θ1 (panta liniei, cât crește y pentru fiecare unitate în plus în X)
print("Intercept (θ0):", model.intercept_)
print("Slope (θ1):", model.coef_[0])

new_houses = np.array([
    880, 990, 870, 1025, 890, 1115, 1200, 980, 1255, 1320,
    1185, 1400, 1505, 1370, 1600, 1720
]).reshape(-1, 1)

# Predicțiile modelului
predictions = model.predict(new_houses)

# Afișăm toate predicțiile
for size, price in zip(new_houses.flatten(), predictions):
    print(f"Predicted price for house {size} ft^2: ${price*1000:.0f}")

# Explicație:
# - model.predict() aplică formula hθ(x) = θ0 + θ1*x pentru fiecare x
# - Obținem estimări pentru preț pe baza liniei de regresie

# --- 6. Evaluarea performanței modelului ---
# y_pred = predicțiile modelului pentru datele originale
y_pred = model.predict(X)

# Mean Squared Error (MSE) = media pătratelor diferențelor dintre valorile reale și cele prezise
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Explicație:
# - MSE = (1/m) * Σ(hθ(xi) - yi)^2
# - Cu cât MSE este mai mic, cu atât linia de regresie se potrivește mai bine datelor

# --- 7. Vizualizarea datelor și a liniei de regresie ---
plt.scatter(X, y, color='blue', label='Actual prices')  # punctele reale
plt.plot(X, y_pred, color='red', label='Regression line')  # linia de regresie
plt.scatter(new_houses, predictions, color='green', label='Predicted prices')  # predicțiile pentru case noi
plt.xlabel('Size (ft^2)')  # etichetă axa X
plt.ylabel('Price ($1000)')  # etichetă axa Y
plt.title('Linear Regression: House Prices vs Size')  # titlul graficului
plt.legend()  # afișează legenda
plt.show()  # afișează graficul
