import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1. Date simple (house prices in Portland)
# ================================================================
# X = dimensiunea casei (independent variable / feature)
# y = prețul casei (dependent variable / target)
# În sklearn, X trebuie să fie 2D (n_samples x n_features)
X = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500]).reshape(-1, 1)
y = np.array([150, 180, 200, 250, 300, 330, 360, 400])

# ================================================================
# 2. Pregătirea pentru gradient descent
# ================================================================
# Adăugăm o coloană de 1 pentru intercept (θ0)
# În sklearn, interceptul poate fi calculat automat cu fit_intercept=True
# Dar aici îl includem manual pentru a vedea cum gradient descent îl optimizează
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (n_samples, 2)

# ================================================================
# 3. Parametrii gradient descent
# ================================================================
# θ = vector de coeficienți [θ0, θ1]
theta = np.random.randn(2,1)  # inițializare aleatorie
learning_rate = 0.0000001      # α - learning rate, mic pentru stabilitate
n_iterations = 2000             # număr de pași
m = len(X_b)                    # număr de eșantioane

y = y.reshape(-1,1)  # reshape pentru compatibilitate matricială

# ================================================================
# 4. Funcția cost (Mean Squared Error)
# ================================================================
# În sklearn, funcția cost internă pentru LinearRegression este MSE
# J(θ0,θ1) = 1/(2m) * Σ(hθ(xi) - yi)^2
def compute_cost(X_b, y, theta):
    predictions = X_b.dot(theta)  # hθ(x) = θ0 + θ1*x
    errors = predictions - y
    return (1/(2*m)) * np.sum(errors**2)

# ================================================================
# 5. Gradient descent (optimizare coeficienți)
# ================================================================
# Ideea: actualizăm θ0 și θ1 pas cu pas, pentru a minimiza MSE
# Acest proces este similar cu ce face sklearn intern când folosește Least Squares
for iteration in range(n_iterations):
    gradients = (1/m) * X_b.T.dot(X_b.dot(theta) - y)  # derivată parțială ∂J/∂θ
    theta = theta - learning_rate * gradients           # actualizare coeficienți

# ================================================================
# 6. Rezultatele optimizării
# ================================================================
print("θ0 (intercept):", theta[0,0])  # interceptul liniei de regresie
print("θ1 (slope):", theta[1,0])      # panta liniei de regresie
print("Final cost (MSE):", compute_cost(X_b, y, theta))

# Comentariu sklearn:
# - În sklearn, .fit(X, y) ar fi rezolvat automat această optimizare, calculând θ0 și θ1
# - Aici demonstrăm conceptul matematic din spatele funcției .fit()

# ================================================================
# 7. Predicții pentru noi date (similar cu .predict() în sklearn)
# ================================================================
X_new = np.array([1600, 3200]).reshape(-1,1)
X_new_b = np.c_[np.ones((X_new.shape[0],1)), X_new]
predictions = X_new_b.dot(theta)  # hθ(x_new)
for size, price in zip(X_new.flatten(), predictions.flatten()):
    print(f"Predicted price for house {size} ft^2: ${price*1000:.0f}")

# Comentariu sklearn:
# - În sklearn, .predict(X_new) folosește coeficienții calculați de .fit()
# - Obținem estimări pentru valorile de intrare noi

# ================================================================
# 8. Vizualizare
# ================================================================
plt.scatter(X, y, color='blue', label='Actual prices')       # date reale
plt.plot(X, X_b.dot(theta), color='red', label='Regression line (GD)')  # linia optimizată
plt.scatter(X_new, predictions, color='green', label='Predicted prices') # predicții noi
plt.xlabel('House size (ft^2)')
plt.ylabel('Price ($1000)')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()

# ================================================================
# Comentarii generale legate de sklearn
# ================================================================
# 1. LinearRegression() din sklearn folosește intern o formulă de tip Least Squares
#    care găsește coeficienții θ0 și θ1 care minimizează MSE
# 2. .fit(X, y) = procesul de optimizare, similar cu gradient descent
# 3. .predict(X_new) = hθ(X_new), folosind coeficienții calculați
# 4. mean_squared_error(y, y_pred) = funcția cost evaluată pentru performanță
# 5. Gradient descent oferă o metodă de optimizare generală, folosită și în ML mai complex
