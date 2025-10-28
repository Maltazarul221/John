# =======================================================
# Exploratory Data Analysis & Data Preprocessing Exercises
# =======================================================

# =======================================================
# Importăm bibliotecile necesare
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_clipping():
    fig, axes = plt.subplots(1, 3, figsize=(7, 5))
    axes = axes.flatten()
    #„salariu” → include outlierii inițiali ( cu mustăți lungi)
    #„clipped_salary” → are marginile taiate (limitate la percentila 5% și 95%)

    sns.boxplot(data=df[["salariu", "clipped_salary"]], ax=axes[0])
    axes[0].set_title("Comparatie Boxplot: inainte vs dupa Clipping")
    axes[0].set_ylabel("Valoare salariu")
    axes[0].set_xlabel("Set de date")


    sns.histplot(df["salariu"], color="skyblue", label="Inainte", kde=True, ax=axes[1])
    sns.histplot(df["clipped_salary"], color="orange", label="Dupa", kde=True, ax=axes[1])
    axes[1].set_title("Distributia salariilor inainte și după Clipping")
    axes[1].set_xlabel("Salariu")
    axes[1].set_ylabel("Frecventa")
    axes[1].legend()

    min_val = df["salariu"].min()
    max_val = df["salariu"].max()
    print(min_val, max_val)
    axes[2].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Diagonala y = x")

    axes[2].scatter(df["salariu"], df["clipped_salary"], color="teal", alpha=0.6)
    axes[2].set_title("Comparatie pct cu pct: inainte si după clipping")
    axes[2].set_xlabel("Salariu original")
    axes[2].set_ylabel("Salariu limitat")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()

def display_statistics():
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    axes = axes.flatten()

    bars = sns.barplot(x=grupare.index, y="numar_angajati", data=grupare, color="skyblue", ax=axes[0])
    axes[0].set_title("Numar de angajati pe departament")
    axes[0].set_xlabel("Departament")
    axes[0].set_ylabel("Numar angajati")
    # axes[0].bar_label(bars, fmt='%.2f', label_type='edge', padding=3)
    axes[0].set_ylim(0, grupare["numar_angajati"].max() * 1.2)


    # # Adaugă etichete deasupra barelor
    for i, v in enumerate(grupare["numar_angajati"]):
        axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom')


    axes[1].pie(
        grupare["numar_angajati"],
        labels=grupare.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"}
    )
    axes[1].set_title("Proportia angajatilor per departament")

def display_feature_extraction():
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    axes = axes.flatten()
    #Fiecare bara = un angajat.
    #O bara inalta → salariu mare raportat la experienta mica.
    #O bara mica → salariu proportional cu experienta.

    sns.barplot(x=df["id"], y=df["raportul 2"], color="orange", ax=axes[0])
    axes[0].set_title("Raportul Salariu / Experienta pe angajat")
    axes[0].set_xlabel("ID angajat")
    axes[0].set_ylabel("Raport (Salariu / Experienta)")
    axes[0].grid(axis="y", alpha=0.3)


    sns.boxplot(y=df["raportul 2"], color="skyblue", ax=axes[1])
    axes[1].set_title("Distributia valorilor raportului Salariu / Experienta")
    axes[1].set_ylabel("Raport (Salariu / Experienta)")

# =======================================================
# Date inițiale
# =======================================================
# Creăm un DataFrame Pandas cu informații despre angajați
data = {
    "id": range(1, 11),
    "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
    "salariu": [4500., 7200., 6800., 5200., 4700., 6000., 5100., 7500., 6400., 5600.],
    "vârstă": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
    "experiență_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
}

# Transformăm dicționarul într-un DataFrame
df = pd.DataFrame(data)

# Afișăm datele pentru a verifica structura
print("=== Datele inițiale ===")
print(df)

# =======================================================
# 1️⃣ Handling Missing Data (valori lipsă)
# =======================================================
# CERINȚĂ:
# Simulează valori lipsă pentru 'salariu' și 'vârstă' și înlocuiește-le cu mediana coloanei.

df.loc[1,"salariu"] = np.nan                                  # accesam linia 1, coloana salariu prin loc - liniile necesita index; np.nan strica
df.loc[4,"vârstă"] = np.nan

print(df.isna().sum())                                        # isna function returns a mask and needs sum to add the values

df["salariu"] = df["salariu"].fillna(df["salariu"].median())  # pune mediana in locul valorilor lipsa; mediana e o functie similara cu mean
df["vârstă"] = df["vârstă"].fillna(df["vârstă"].median())     #

print(df)
print(df.isna().sum())

# =======================================================
# 2️⃣ Normalization (Min-Max Scaling)
# Formula matematica pentru normalizare (x - min) / (max - min)
# min o valoare de 0 si una de 1 (min si max) dar putem avea mai multe daca valorile sunt egale
# =======================================================
# CERINȚĂ:
# Normalizează coloana 'salariu' astfel încât valorile să fie între 0 și 1.

df["salariu_normalizat"] = (df["salariu"] - df["salariu"].min()) / (df["salariu"].max() - df["salariu"].min())
print(df["salariu_normalizat"])

# Tema: afisati pe grafic ceva semnificativ pentru normalizare (poate 2 charts)

# =======================================================
# 3️⃣ Standardization / Scalare (Z-score)
# Formula: (x - media) / deviatia_standard
# =======================================================
# CERINȚĂ:
# Standardizează coloana 'experiență_ani' astfel încât să aibă media 0 și deviația standard 1.

df["experienta_standardizata"] = (df["experiență_ani"] - df["experiență_ani"].mean()) / df["experiență_ani"].std()
print(df[["experiență_ani","experienta_standardizata"]])
print(df["experiență_ani"].mean())

# Tema: afisati pe grafic ceva semnificativ

# =======================================================
# 4️⃣ Discretization (Binning)
# =======================================================
# CERINȚĂ:
# Împarte coloana 'vârstă' în 3 categorii: 'Tânăr', 'Matur', 'Senior'

labels = ["Tanar", "Matur", "Senior"]                       # etichete sub format de string
bins = [0, 30, 40, 130]                                     # 4 numere care indica limitele intervalelor (3 intervale asociate cu label)

df["categorie_varsta"] = pd.cut(
    df["vârstă"],
    bins= bins,
    labels=labels,
    include_lowest=True                                    # by default daca nu punem e false
)

print(df[["vârstă", "categorie_varsta"]])
print(df["categorie_varsta"].value_counts())

# Tema: afisati pe grafic ceva semnificativ + veniti cu alte metode de calcul

# =======================================================
# 5️⃣ Categorical Imputation (completare categorii lipsă)
# =======================================================
# CERINȚĂ:
# Simulează o valoare lipsă în coloana 'departament' și completează cu cel mai frecvent departament

df.loc[3,"departament"] = np.nan                                             # accesam linia 1, coloana salariu prin loc - liniile necesita index; np.nan strica

print(df.isna().sum())                                                       # isna function returns a mask and needs sum to add the values

# most_frequent = df["departament"].mode()[0]

# df["departament"] = df["departament"].fillna("IT")                         # pune "IT" in locul valorilor lipsa; mediana e o functie similara cu mean
df["departament"] = df["departament"].fillna(df["departament"].mode()[0])    # df["departament"].mode() => returnează un series / array cu valorile cele mai frecvente in ordine,
                                                                             # [0] => ia prima valoare (în caz că sunt mai multe la egalitate)

print("\n După completare:\n", df)
print("\n Valori lipsa ramase:\n", df.isna().sum())

# =======================================================
# 6️⃣ Noise Removal (Outlier detection = eliminarea valorilor prea departe de medie, > x deviatii)
# Formula: Detectarea se face comparând fiecare valoare cu media +/- factor * deviația standard. | factor = numarul de deviatii pentru utliers (2/3)
# =======================================================
# CERINȚĂ:
# Identifică eventualii outlier-i în coloana 'salariu' folosind deviația standard.

media_salarii = df["salariu"].mean()
deviatia_std = df["salariu"].std()

factor = 1

lower_limit = media_salarii - factor * deviatia_std
upper_limit = media_salarii + factor * deviatia_std

print(f"Limită inferioară: {lower_limit}")
print(f"Limită superioară: {upper_limit}")

outliers = df[(df["salariu"] <= lower_limit) | (df["salariu"] > upper_limit)]  # | = or in pandas
print("\n Outliers:")
print(outliers)

# 0 <= x <= 10      x e intre 0 si 10
# lower_limit <= df["salariu"] <= upper_limit

masca = (df["salariu"] >= lower_limit) & (df["salariu"] <= upper_limit)       # & = and in pandas; generam masca

df_no_outliers = df[masca]                                                    # aplicam masca
print("\n După completare:\n", df)
print("\n Valori lipsa ramase:\n", df_no_outliers)

# df_no_outliers = df[(df["salariu"] >= lower_limit) & (df["salariu"] <= upper_limit)]


# =======================================================
# 7️⃣ Limiting Outliers (Clipping)
# =======================================================
# CERINȚĂ:
# În loc să eliminăm outlier-ii, putem să îi limităm la anumite valori maxime/minime (percentile).

lower_limit = df["salariu"].quantile(0.05)
upper_limit = df["salariu"].quantile(0.95)

df["clipped_salary"] = df["salariu"].clip(lower = lower_limit, upper = upper_limit)

print("\n 7. Inainte si dupa clipping:\n", df[["salariu","clipped_salary"]])
display_clipping()

# =======================================================
# 8️⃣ Grouping și Aggregation
# =======================================================
# CERINȚĂ:
# Grupăm datele după 'departament' și calculăm statistici agregate.

grupare = df.groupby("departament").agg(
    numar_angajati = ("id","count")
)

print("\n 8. Grouping și Aggregation :\n", grupare)

display_statistics()

# =======================================================
# 9️⃣ Feature Extraction simplu
# =======================================================
# CERINȚĂ:
# Creăm o nouă caracteristică: raportul salariu/experiență

df.loc[3,"experiență_ani"] = np.nan                                         # adds NaN values

df["experiență_ani"] = df["experiență_ani"].replace(0,np.nan)               # avoids 0 division errors

df["raportul"] = df["salariu"] / df["experiență_ani"]    # calculates

df.loc[4,"experiență_ani"] = 0                                              # adds 0 values

df["raportul 2"] = np.where(
    df["experiență_ani"] != 0,
    df["salariu"] / df["experiență_ani"],
    0                                                                       # default value (if /0)
)

print("\n 9. Feature Extraction simplu:\n", df[["salariu", "experiență_ani","raportul", "raportul 2"]])
display_feature_extraction()



# =======================================================
# 1️⃣0️⃣ DataFrame final
# =======================================================
# Afișăm DataFrame-ul complet după toate transformările și exercițiile de preprocessing

plt.show()