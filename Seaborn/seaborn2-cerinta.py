# =======================================================
# Vizualizări cu Seaborn pentru date angajați (subplot)
# =======================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =======================================================
# Date angajați
# =======================================================
data = {
    "id": range(1, 11),
    "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
    "salariu": [4500., 7200., 6800., 5200., 4700., 6000., 5100., 7500., 6400., 5600.],
    "varsta": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
    "experienta_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
}
df = pd.DataFrame(data)

# =======================================================
# CERINȚE:
# =======================================================
# 1. BOX PLOT – Salariu pe departament           ----- (distributie grupata -- outlier = valoare indepartata)
#   - Vizualizează distribuția salariilor pe departamente.
#   - Observă medianele și eventualele valori extreme.

plt.figure()
plt.title("Distribuția salariilor pe departamente")
x = sns.boxplot(            # e mai indicat sa pui totul intr-o variabila
    data = df,        # data contine df-ul cu datele, de unde isi ia informatiile
    x = "departament",           # coloanele din df
    y = "salariu",           # coloanele din df
    hue = "departament",         # coloana dupa care coloreaza diferit (automat)
    palette= "pastel",          # saturatia
    legend = True,
    #loc = "lower right"        - see comment below
)
x.legend(           # pentru a schimba legenda (sau probabil si alte chestii) se pune plt-ul in variabila
    title = "Departament",
    loc = "lower right",
    fontsize = 12,
    ncol = 4,
    frameon = False
)
plt.xlabel("Departament")
plt.ylabel("Salariu")

# =======================================================
# 2. VIOLIN PLOT – Distribuția vârstei
#   - Afișează distribuția vârstelor pe departamente.
#   - Observă densitatea și intervalele în care se află majoritatea angajaților.

plt.figure(figsize = (5, 5))
plt.title("Distributia varstei pe departament")
sns.violinplot(
    data = df,
    x = "departament",
    y = "varsta",
    hue = "departament",
    legend = True,
    inner = "point",
)

# =======================================================
# 3. HEATMAP – Corelația între variabile
#   - Calculează matricea de corelație între salariu, vârstă și experiență.
#   - Afișează valorile și observă ce variabile sunt corelate pozitiv sau negativ.

df_numeric = df[['salariu', 'varsta', 'experienta_ani']]
corr = df_numeric.corr()        # .corr = corelatie intre valori
plt.figure(figsize = (5,5))
plt.title("Corelatia intre variabile")
sns.heatmap(
    data = corr,        # correlation
    annot = True,       # annotation
    cmap = "coolwarm",      # coolwarm = cool, valori mici - warm , valori mari
    fmt = ".2f"          # format la nr cu decimale
)

# =======================================================
# 4. KDE PLOT – Densitatea salariilor           ------ distributie = KDE sau histograma
#   - Creează un grafic KDE pentru a vizualiza distribuția salariilor.
#   - Observă intervalele cu cea mai mare densitate (unde sunt concentrați angajații).

plt.figure(figsize = (5, 5))
plt.title("Distributia salariilor")
sns.kdeplot(
    data = df,
    x = "salariu",          # se da o axa (x) si in functie de asta isi alege singur "densitatea"
    fill = True,
    #cmap = "Blues"
    color = "#00ccff"       # si cmap si color functioneaza la fel
)

plt.show()