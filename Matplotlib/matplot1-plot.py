import matplotlib.pyplot as plt   # Importă modulul pyplot din biblioteca matplotlib (folosit pentru desenarea graficelor)
import numpy as np                # Importă biblioteca numpy (utilă pentru calcule numerice și lucrul cu array-uri)

# ==============================================
# Date fictive ML
# ==============================================
epoci = np.arange(1, 11)   # Creează un array cu valorile 1,2,...,10 — fiecare valoare reprezintă o epocă (iterare în antrenare)
train_accuracy = np.array([0.6, 0.68, 0.72, 0.78, 0.82, 0.85, 0.88, 0.9, 0.91, 0.93])  # Acuratețea pe setul de antrenare pentru fiecare epocă
val_accuracy = np.array([0.58, 0.63, 0.69, 0.74, 0.78, 0.81, 0.83, 0.84, 0.85, 0.86])   # Acuratețea pe setul de validare
train_loss = np.array([1.2, 0.9, 0.75, 0.6, 0.5, 0.45, 0.38, 0.35, 0.32, 0.3])         # Valori de pierdere (loss) pentru train
val_loss = np.array([1.3, 1.0, 0.85, 0.7, 0.6, 0.55, 0.5, 0.48, 0.45, 0.43])           # Valori de pierdere pentru validation
numar_clienti = np.array([20, 25, 30, 35, 40, 50])  # Date exemplu — număr de clienți
profit = np.array([30, 40, 50, 60, 70, 90])         # Profit corespunzător numărului de clienți

classes = ['Clasa A', 'Clasa B', 'Clasa C']  # Numele celor 3 clase
predictions = [50, 30, 20]                   # Numărul de predicții pentru fiecare clasă
scores_clasa_A = np.random.beta(a=2, b=5, size=100)  # Generează 100 valori aleatoare dintr-o distribuție Beta (forma scorurilor)
cum_correct_train = np.cumsum(train_accuracy*10)     # Suma cumulativă (total parțial) a acurateței, multiplicată cu 10

# ==============================================
# 1. Line plot: Acuratețe Train vs Validation
plt.figure(figsize=(7,5))
# Creează o figură nouă pentru grafic, cu dimensiunea 7 inch lățime și 5 inch înălțime

plt.plot(epoci, train_accuracy, marker='o', label='Train')
# Desenează o linie cu:
#  - epoci: valorile de pe axa X (1,2,...,10)
#  - train_accuracy: valorile de pe axa Y
#  - marker='o': marchează fiecare punct cu un cerc
#  - label='Train': eticheta pentru legendă

plt.plot(epoci, val_accuracy, marker='s', label='Validation')
# A doua linie pe același grafic:
#  - marker='s': fiecare punct e un pătrat
#  - label='Validation': va apărea în legendă

plt.title('Acuratețe Train vs Validation')  # Titlul graficului
plt.xlabel('Epoca')                         # Numele axei X
plt.ylabel('Acuratețe')                     # Numele axei Y
plt.ylim(0, 1)                              # Limite axa Y între 0 și 1 (pentru valori de acuratețe)
plt.legend()                                # Afișează legenda (Train / Validation)
plt.show()                                  # Afișează efectiv graficul pe ecran

# ==============================================
# 2. Line plot: Pierdere Train vs Validation
plt.figure(figsize=(7,5))
# Creează o figură nouă pentru graficul de pierdere (loss)

plt.plot(epoci, train_loss, marker='o', label='Train Loss')
# Linie pentru pierderea la antrenare:
#  - X: epoci
#  - Y: valorile train_loss
#  - marker='o': puncte în formă de cerc
#  - label: nume pentru legendă

plt.plot(epoci, val_loss, marker='s', label='Validation Loss')
# Linie pentru pierderea la validare:
#  - marker='s': pătrate
#  - label='Validation Loss': numele afișat în legendă

plt.title('Pierdere Train vs Validation')  # Titlul graficului
plt.xlabel('Epoca')                        # Eticheta axei X
plt.ylabel('Loss')                         # Eticheta axei Y
plt.legend()                               # Afișează legenda
plt.show()                                 # Afișează graficul

# ==============================================
# 3. Bar plot: Predicții pe clase
plt.figure()
# Creează o figură nouă

plt.bar(classes, predictions, color=['blue','green','orange'])
# Desenează un grafic cu bare verticale:
#  - classes: valorile axei X (numele claselor)
#  - predictions: valorile axei Y (număr predicții)
#  - color: lista de culori pentru fiecare bară

plt.title('Număr predicții pe clasă')  # Titlu
plt.xlabel('Clasă')                    # Eticheta axei X
plt.ylabel('Număr predicții')          # Eticheta axei Y
plt.show()                             # Afișează graficul

# ==============================================
# 4. Horizontal Bar plot
plt.figure()
# Creează o figură nouă

plt.barh(classes, predictions, color=['blue','green','orange'])
# Desenează un grafic cu bare orizontale:
#  - classes: valorile de pe axa Y
#  - predictions: lungimea barelor
#  - color: culorile barelor

plt.title('Număr predicții pe clasă (orizontal)')  # Titlu
plt.xlabel('Număr predicții')                      # Eticheta axei X
plt.ylabel('Clasă')                                # Eticheta axei Y
plt.show()                                         # Afișează graficul

# ==============================================
# 5. Histogram: Distribuția scorurilor Clasa A
plt.figure()
# Creează o figură nouă

plt.hist(scores_clasa_A, bins=10, color='skyblue', edgecolor='black')
# Creează un histogramă:
#  - scores_clasa_A: datele folosite
#  - bins=10: împarte datele în 10 intervale
#  - color='skyblue': culoarea barelor
#  - edgecolor='black': contur negru la bare

plt.title('Distribuția scorurilor Clasa A')  # Titlu
plt.xlabel('Probabilitate')                  # Eticheta axei X
plt.ylabel('Număr exemple')                  # Eticheta axei Y
plt.show()                                   # Afișează graficul

# ==============================================
# 6. Pie chart: Proporția claselor prezise
plt.figure()
# Creează o figură nouă

plt.pie(predictions, labels=classes, autopct='%1.1f%%', startangle=90, colors=['blue','green','orange'])
# Creează o diagramă circulară:
#  - predictions: valorile pentru fiecare sector
#  - labels=classes: numele claselor
#  - autopct='%1.1f%%': afișează procentul cu o zecimală
#  - startangle=90: pornește de la 90° pentru a fi aliniat
#  - colors: culorile sectoarelor

plt.title('Proporția claselor prezise')  # Titlu
plt.show()                               # Afișează graficul

# ==============================================
# 7. Area plot: Evoluția acurateței cumulative
plt.figure()
# Creează o figură nouă

plt.fill_between(epoci, train_accuracy, color="skyblue", alpha=0.5, label='Train')
# Umple zona sub curba train_accuracy:
#  - epoci: X
#  - train_accuracy: Y
#  - color='skyblue': culoare
#  - alpha=0.5: transparență
#  - label='Train': etichetă pentru legendă

plt.plot(epoci, train_accuracy, color="Slateblue", alpha=0.8)
# Trasează linia peste zona umplută:
#  - color='Slateblue': culoarea liniei
#  - alpha=0.8: opacitate

plt.fill_between(epoci, val_accuracy, color="lightgreen", alpha=0.5, label='Validation')
plt.plot(epoci, val_accuracy, color="green", alpha=0.8)
# Face același lucru pentru valorile de validare

plt.title('Evoluția acurateței cumulative')  # Titlu
plt.xlabel('Epoca')                          # Etichetă X
plt.ylabel('Acuratețe')                      # Etichetă Y
plt.legend()                                 # Legendă
plt.show()                                   # Afișează graficul

# ==============================================
# 8. Step plot: Cumulul exemplelor corect clasificate
plt.figure()
# Creează o figură nouă

plt.step(epoci, cum_correct_train, where='mid', color='brown')
# Creează un grafic tip scară:
#  - epoci: X
#  - cum_correct_train: Y
#  - where='mid': pașii se schimbă la mijlocul intervalului
#  - color='brown': culoarea liniei

plt.title('Exemple corect clasificate cumulative')  # Titlu
plt.xlabel('Epoca')                                # Etichetă X
plt.ylabel('Exemple corecte')                      # Etichetă Y
plt.show()                                         # Afișează graficul

# ==============================================
# 9. Stack plot: Contribuția Train/Validation
plt.figure()
# Creează o figură nouă

plt.stackplot(epoci, train_accuracy*10, val_accuracy*10, labels=['Train','Validation'], colors=['blue','green'])
# Creează un grafic de tip "stacked area":
#  - epoci: X
#  - train_accuracy*10, val_accuracy*10: două serii de date suprapuse
#  - labels: etichete pentru fiecare serie
#  - colors: culori pentru zone

plt.title('Contribuția Train/Validation la predicții corecte')
plt.xlabel('Epoca')
plt.ylabel('Exemple corect clasificate')
plt.legend(loc='upper left')  # Afișează legenda în colțul stânga-sus
plt.show()

# ==============================================
# COMBINATII
# ==============================================

np.random.seed(42)  # Fixează generatorul aleator pentru rezultate reproductibile

n_samples = 300                                   # Număr de eșantioane generate
prob_pred = np.random.rand(n_samples)             # Probabilități prezise (valori între 0 și 1)
true_label = np.random.binomial(1, 0.5, size=n_samples)  # Etichete reale 0 sau 1 generate aleator

true_values = np.random.normal(loc=50, scale=10, size=n_samples)  # Valori reale (distribuție normală)
pred_values = true_values + np.random.normal(loc=0, scale=5, size=n_samples)  # Valori prezise cu zgomot adăugat

plt.figure(figsize=(7,5))
plt.scatter(true_label + 0.02*np.random.randn(n_samples), prob_pred, alpha=0.5, color='blue')
# Creează un scatter plot:
#  - X: eticheta reală (0 sau 1) ușor „dispersată” aleator pentru vizibilitate
#  - Y: probabilitatea prezisă
#  - alpha=0.5: transparență
#  - color='blue': culoare puncte

plt.title('Probabilitate prezisă vs etichetă reală')
plt.xlabel('Etichetă reală (0 sau 1)')
plt.ylabel('Probabilitate prezisă')
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(true_values, pred_values, alpha=0.5, color='green')
# Scatter pentru regresie:
#  - X: valori reale
#  - Y: valori prezise
#  - alpha=0.5: transparență
#  - color='green': culoarea punctelor

plt.plot([30, 70], [30, 70], color='red', linestyle='--', label='Predicție perfectă')
# Linie de referință y=x (predicție perfectă):
#  - [30,70]: interval X și Y
#  - color='red': culoarea liniei
#  - linestyle='--': linie întreruptă
#  - label: etichetă pentru legendă

plt.title('Valori reale vs Valori prezise (regresie)')
plt.xlabel('Valori reale')
plt.ylabel('Valori prezise')
plt.legend()
plt.show()
