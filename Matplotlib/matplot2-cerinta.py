import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dataset
data = {
    "id": range(1, 11),
    "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
    "salariu": [4500., 7200., 6800., 5200., 4700., 6000., 5100., 7500., 6400., 5600.],
    "vârstă": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
    "experiență_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
}
df = pd.DataFrame(data)

# ==============================================
# Cerințe grafice:
# ==============================================
# 1. Line plot: salariu mediu acumulat pe angajați
final_data = df["salariu"].cumsum()
plt.figure()
plt.title("Salariu acumulat pe angajați")
plt.xlabel("Index angajat")
#plt.xlim(0,100)
#plt.ylim(0,10000)
plt.ylabel("salariu cumulat")
plt.plot(df.index, final_data , marker = "s" , color = "#78ed09", label = "salariu_c")
plt.plot(df.index, df['salariu'] , marker = "o" , color = (0.5, 0.5, 0.5) , label = "salariu")
plt.legend()
plt.show()


# # Exemplu de date
# epoci = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# train_accuracy = np.array([0.2, 0.35, 0.5, 0.6, 0.68, 0.74, 0.79, 0.83, 0.86, 0.88])
# val_accuracy   = np.array([0.18, 0.3, 0.45, 0.55, 0.63, 0.69, 0.73, 0.77, 0.8, 0.82])
# # Creează o linie "exponențială" simplă prin np.exp
# # (normalizăm pentru a rămâne între 0 și 1)
# train_curve = 1 - np.exp(-0.3*epoci)  # crește rapid și se stabilizează
# val_curve   = 1 - np.exp(-0.28*epoci)
#
# final_data = df["salariu"].cumsum()
# plt.figure()
# plt.title("Salariu acumulat pe angajați")
# plt.xlabel("Index angajat")
# #plt.xlim(0,100)
# #plt.ylim(0,10000)
# plt.ylabel("salariu cumulat")
# plt.plot(df.index, train_curve , marker = "s" , color = "green", label = "salariu_c")
# plt.plot(df.index,  val_curve, marker = "o" , color = "blue" , label = "salariu")
# plt.legend()
# plt.show()

# 2. Bar plot: salariu mediu pe departament
salariu_mediu = df.groupby(['departament'])['salariu'].mean()
plt.figure()
plt.title("Salariu mediu pe departament")
plt.xlabel("Departament")
plt.ylabel("Salariu mediu pe DPT")
plt.bar(salariu_mediu.index, salariu_mediu.values)
plt.legend()
plt.show()

# 3. Horizontal bar plot: experiență medie pe departament
# 4. Scatter plot: experiență vs salariu
# 5. Histogram: distribuția salariilor
# 6. Pie chart: proporția angajaților pe departamente
# 7. Area plot: salariu cumulativ pe angajați
# 8. Step plot: salariu cumulativ
# 9. Stack plot: contribuția fiecărui departament la salariul total
# 10. Subplots: scatter experiență vs salariu și vârstă vs salariu
