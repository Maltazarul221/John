import matplotlib.pyplot as plt
import numpy as np

# ==============================================
# Date fictive ML
# ==============================================
np.random.seed(42)
epoci = np.arange(1, 11)
train_accuracy = np.array([0.6, 0.68, 0.72, 0.78, 0.82, 0.85, 0.88, 0.9, 0.91, 0.93])
val_accuracy   = np.array([0.58, 0.63, 0.69, 0.74, 0.78, 0.81, 0.83, 0.84, 0.85, 0.86])
train_loss = np.array([1.2, 0.9, 0.75, 0.6, 0.5, 0.45, 0.38, 0.35, 0.32, 0.3])
val_loss   = np.array([1.3, 1.0, 0.85, 0.7, 0.6, 0.55, 0.5, 0.48, 0.45, 0.43])
numar_clienti = np.array([20, 25, 30, 35, 40, 50])
profit = np.array([30, 40, 50, 60, 70, 90])
classes = ['Clasa A', 'Clasa B', 'Clasa C']
predictions = [50, 30, 20]
scores_clasa_A = np.random.beta(a=2, b=5, size=100)
cum_correct_train = np.cumsum(train_accuracy*10)
n_samples = 300
prob_pred = np.random.rand(n_samples)
true_label = np.random.binomial(1, 0.5, size=n_samples)
true_values = np.random.normal(loc=50, scale=10, size=n_samples)
pred_values = true_values + np.random.normal(loc=0, scale=5, size=n_samples)

# ==============================================
# Creare subplot grid
# ==============================================
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()  # pentru acces ușor cu index

# 1. Line plot: Acuratețe
axes[0].plot(epoci, train_accuracy, marker='o', label='Train')
axes[0].plot(epoci, val_accuracy, marker='s', label='Validation')
axes[0].set_title('Acuratețe Train vs Validation')
axes[0].set_xlabel('Epoca')
axes[0].set_ylabel('Acuratețe')
axes[0].legend()
axes[0].set_ylim(0, 1)

# 2. Line plot: Pierdere
axes[1].plot(epoci, train_loss, marker='o', label='Train Loss')
axes[1].plot(epoci, val_loss, marker='s', label='Validation Loss')
axes[1].set_title('Pierdere Train vs Validation')
axes[1].set_xlabel('Epoca')
axes[1].set_ylabel('Loss')
axes[1].legend()

# 3. Bar plot
axes[2].bar(classes, predictions, color=['blue','green','orange'])
axes[2].set_title('Predicții pe clase')
axes[2].set_xlabel('Clasă')
axes[2].set_ylabel('Număr predicții')

# 4. Horizontal bar plot
axes[3].barh(classes, predictions, color=['blue','green','orange'])
axes[3].set_title('Predicții pe clase (orizontal)')
axes[3].set_xlabel('Număr predicții')
axes[3].set_ylabel('Clasă')

# 5. Histogram
axes[4].hist(scores_clasa_A, bins=10, color='skyblue', edgecolor='black')
axes[4].set_title('Distribuția scorurilor Clasa A')
axes[4].set_xlabel('Probabilitate')
axes[4].set_ylabel('Număr exemple')

# 6. Pie chart
axes[5].pie(predictions, labels=classes, autopct='%1.1f%%', startangle=90, colors=['blue','green','orange'])
axes[5].set_title('Proporția claselor prezise')

# 7. Area plot
axes[6].fill_between(epoci, train_accuracy, color="skyblue", alpha=0.5, label='Train')
axes[6].plot(epoci, train_accuracy, color="Slateblue", alpha=0.8)
axes[6].fill_between(epoci, val_accuracy, color="lightgreen", alpha=0.5, label='Validation')
axes[6].plot(epoci, val_accuracy, color="green", alpha=0.8)
axes[6].set_title('Acuratețe cumulative')
axes[6].set_xlabel('Epoca')
axes[6].set_ylabel('Acuratețe')
axes[6].legend()

# 8. Step plot
axes[7].step(epoci, cum_correct_train, where='mid', color='brown')
axes[7].set_title('Exemple corect clasificate cumulative')
axes[7].set_xlabel('Epoca')
axes[7].set_ylabel('Exemple corecte')

# 9. Stack plot
axes[8].stackplot(epoci, train_accuracy*10, val_accuracy*10, labels=['Train','Validation'], colors=['blue','green'])
axes[8].set_title('Contribuția Train/Validation')
axes[8].set_xlabel('Epoca')
axes[8].set_ylabel('Exemple corect clasificate')
axes[8].legend(loc='upper left')

# 10. Scatter plot: probabilitate vs etichetă
axes[9].scatter(true_label + 0.02*np.random.randn(n_samples), prob_pred, alpha=0.5, color='blue')
axes[9].set_title('Probabilitate vs etichetă reală')
axes[9].set_xlabel('Etichetă')
axes[9].set_ylabel('Probabilitate prezisă')

# 11. Scatter plot: valori reale vs prezise
axes[10].scatter(true_values, pred_values, alpha=0.5, color='green')
axes[10].plot([30, 70], [30, 70], color='red', linestyle='--', label='Perfect')
axes[10].set_title('Valori reale vs prezise')
axes[10].set_xlabel('Valori reale')
axes[10].set_ylabel('Valori prezise')
axes[10].legend()

# 12. Placeholder (gol) dacă vrem alte grafice
axes[11].axis('off')  # poate fi folosit pentru alte grafice

plt.tight_layout()
plt.show()
