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
from IPython.core.pylabtools import figsize

def normalize(data_frame, column):
   df_eda = data_frame.copy()
   df_eda[column] = (df_eda[column] - df_eda[column].min()) / (df_eda[column].max() - df_eda[column].min())
   return df_eda

def categorical_imputation(data_frame, column):
    df_eda = data_frame.copy()
    most_freq_value = df_eda[column].mode()[0]
    df_eda[column] = df_eda[column].fillna(most_freq_value)
    return df_eda

def fillna_median(data_frame, column):
    df_eda = data_frame.copy()
    df_eda[column] = df_eda[column].fillna(df_eda[column].median())
    return df_eda

def display_boxplots(df_original, df_eda, states, y):
    plt.figure()
    df_original.loc[:,"stare"] = states[0]
    df_eda.loc[:,"stare"] = states[1]

    df_compare = pd.concat([df_original, df_eda])
    df_compare = df_compare.reset_index(drop=True)
    print(df_compare)

    sns.boxplot(
        data=df_compare,
        x="stare",
        hue="stare",
        y=y,
        palette="pastel"
    )
    plt.title(f"{y}: {states[0]} --> {states[1]}")
    plt.grid(True, linestyle="--", alpha=0.6)

def display_barplots(df_original, df_eda, states, y):
    plt.figure()
    df_original.loc[:,"stare"] = states[0]
    df_eda.loc[:,"stare"] = states[1]

    # print("Value counts")
    original_vc = df_original[y].value_counts()
    eda_vc = df_eda[y].value_counts()
    # print(eda_vc.index, eda_vc.values)
    # print("ORIG")
    # print(original_vc.index, original_vc.values)
    plt.bar(original_vc.index, original_vc.values, label=states[0])
    plt.bar(eda_vc.index, eda_vc.values, label=states[1], bottom=original_vc.values)


    plt.title(f"{y}: {states[0]} --> {states[1]}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()


def display_violinplots(df_original, df_eda, states, y):
    plt.figure()
    df_original.loc[:,"stare"] = states[0]
    df_eda.loc[:,"stare"] = states[1]

    df_compare = pd.concat([df_original, df_eda])
    df_compare = df_compare.reset_index(drop=True)
    print(df_compare)

    sns.violinplot(
        data=df_compare,
        x="stare",
        y=y,
        hue="stare",
        legend=True,
        inner="points",

    )
    plt.title(f"{y}: {states[0]} --> {states[1]}")
    plt.grid(True, linestyle="--", alpha=0.6)

def display_violinplots_twinx(df_original, df_eda, axes, states, y):

    df_original.loc[:,"stare"] = states[0]
    df_eda.loc[:,"stare"] = states[1]

    df_compare = pd.concat([df_original, df_eda])
    df_compare = df_compare.reset_index(drop=True)
    print(df_compare)

    sns.violinplot(
        data=df_original,
        y=y,
        legend=True,
        inner="box",
        palette="Blues",
        hue=None,
        ax=axes,
        alpha=0.5
    )
    axes.set_title(f"{y}: {states[0]} --> {states[1]}")
    axes.grid(True, linestyle="--", alpha=0.6)
    axes2 = axes.twinx()
    sns.violinplot(
        data=df_eda,
        y=y,
        inner="box",
        ax=axes2,
        palette="Reds",
        hue=None,
        alpha=0.5
    )

    axes2.set_ylabel(y, color="darkred")
def display_violinplot_compare(df):

    df_orig = df[["salariu"]].copy()
    df_orig["tip"] = "Salariu original"

    df_norm = df[["salariu_normalizat"]].copy()
    df_norm = df_norm.rename(columns={"salariu_normalizat": "valoare"})
    df_orig = df_orig.rename(columns={"salariu": "valoare"})
    df_norm["tip"] = "Salariu normalizat"


    df_compare = pd.concat([df_orig, df_norm])


    plt.figure(figsize=(6,5))
    sns.violinplot(
        data=df_compare,
        x="tip",
        y="valoare",
        inner="box",
        palette="pastel",
        hue=None
    )

    plt.title("Comparare: Salariu original vs. normalizat")
    plt.ylabel("Valoare")
    plt.grid(True, linestyle="--", alpha=0.6)


def display_normalization():
    fig, axes = plt.subplots(2, 3, figsize=(6,8))
    axes = axes.flatten()
    # axes[0].scatter(df["salariu"], df["salariu_normalizat"], color="seagreen")
    axes[0].plot(df["salariu"], df["salariu_normalizat"], color="seagreen", marker="D")
    axes[0].set_title("Relația dintre salariu original și salariu normalizat")
    axes[0].set_xlabel("Salariu original")
    axes[0].set_ylabel("Salariu normalizat (0–1)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    sns.kdeplot(data=df, x="salariu", fill=True, color="skyblue", linewidth=2, ax=axes[1])
    axes[1].set_xlabel("Salariu")
    axes[1].set_ylabel("Densitate")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Densitatea salarii")

    sns.kdeplot(data=df, x="salariu_normalizat", fill=True, color="skyblue", linewidth=2, ax=axes[2])
    axes[2].set_xlabel("Salariu normalizat")
    axes[2].set_ylabel("Densitate")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Densitatea salarii normalizate")

    corr = df[["salariu", "salariu_normalizat"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=False, ax=axes[3])
    axes[3].set_title("Corelația între salariu și salariu normalizat")

    sns.violinplot(
        data=df,
        y="salariu",
        inner="point",
        color="lightblue",
        ax=axes[4])
    axes[4].set_title("Salariu")
    sns.violinplot(
        data=df,
        y="salariu_normalizat",
        inner="point",
        color="lightblue",
        ax=axes[5])
    axes[5].set_title("Salariu normalizat")
    plt.tight_layout()

def display_standardization():
    fig, axes = plt.subplots(2, 3,  figsize=(6,8))
    axes = axes.flatten()
    # axes[0].scatter(df["salariu"], df["salariu_normalizat"], color="seagreen")
    axes[0].plot(df["experienta_ani"], df["experienta_ani_std"], color="seagreen", marker="D")
    axes[0].set_title("Relația dintre exp ani si exp ani std")
    axes[0].set_xlabel("Experienta ani")
    axes[0].set_ylabel("Experienta ani std")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    sns.kdeplot(data=df, x="experienta_ani", fill=True, color="skyblue", linewidth=2, ax=axes[1])
    axes[1].set_xlabel("Experienta ani")
    axes[1].set_ylabel("Densitate")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Densitatea experienta ani")

    sns.kdeplot(data=df, x="experienta_ani_std", fill=True, color="skyblue", linewidth=2, ax=axes[2])
    axes[2].set_xlabel("Exp ani std")
    axes[2].set_ylabel("Densitate")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Densitate exp ani std ")

    corr = df[["experienta_ani", "experienta_ani_std"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=False, ax=axes[3])
    axes[3].set_title("Corelația între exp ani și exp ani std")

    sns.violinplot(
        data=df,
        y="experienta_ani",
        inner="point",
        color="lightblue",
        ax=axes[4])
    axes[4].set_title("Experienta ani")
    sns.violinplot(
        data=df,
        y="experienta_ani_std",
        inner="point",
        color="lightblue",
        ax=axes[5])
    axes[5].set_title("Experienta ani std")
    plt.tight_layout()

def display_discretization():
    fig, axes = plt.subplots(1, 3)
    axes = axes.flatten()

    sns.kdeplot(data=df, x="varsta", fill=True, color="skyblue", linewidth=2, ax=axes[0])
    axes[0].set_xlabel("Varsta")
    axes[0].set_ylabel("Densitate")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Densitate Varsta")

    # sns.violinplot(
    #     data=df,
    #     x="categorie_varsta",
    #     y="varsta",
    #     inner="point",
    #     color="lightblue",
    #     ax=axes[1])

    # histograma - distribuția reala
    sns.histplot(data=df, x="varsta", bins=10, color="lightgreen", edgecolor="black", ax=axes[1])
    axes[1].set_xlabel("Varsta")
    axes[1].set_ylabel("Frecventa")
    axes[1].set_title("Histograma varsta")
    axes[1].grid(alpha=0.3)

    # barplot - discretizare
    sns.countplot(data=df, x="categorie_varsta", palette="pastel", hue="categorie_varsta", ax=axes[2])
    axes[2].set_xlabel("Categorie varsta")
    axes[2].set_ylabel("Numar persoane")
    axes[2].set_title("Discretizare varsta (Tanar / Matur / Senior)")
    axes[2].grid(alpha=0.3, axis="y")


# =======================================================
# Date inițiale
# =======================================================
# Creăm un DataFrame Pandas cu informații despre angajați
data = {
    "id": range(1, 11),
    "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
    "salariu": [4500., 7200., 6800., 5200., 4700., 6000., 5100., 7500., 6400., 5600.],
    "varsta": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
    "experienta_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
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


df.loc[1, "salariu"] = np.nan
df.loc[7, "salariu"] = np.nan
df.loc[4, "varsta"] = np.nan


display_boxplots(df, fillna_median(df, "salariu"), ("Missing Data", "fillna(median)"), "salariu")
display_boxplots(df, fillna_median(df, "varsta"), ("Missing Data", "fillna(median)"), "varsta")
# display_violinplots(df, fillna_median(df, "salariu"), axes[0], ("Missing Data", "fillna(median)"), "salariu")
# display_violinplots(df, fillna_median(df, "varsta"), axes[1], ("Missing Data", "fillna(median)"), "varsta")

print(df.isna().sum())
df["salariu"] = df["salariu"].fillna(df["salariu"].median())        # pune mediana in locul valorilor lipsa
df["varsta"] = df["varsta"].fillna(df["varsta"].median())
print(df)
print(df.isna().sum())

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru valori lipsa

# =======================================================
# 2️⃣ Normalization (Min-Max Scaling)
# =======================================================
# CERINȚĂ:
# Normalizează coloana 'salariu' astfel încât valorile să fie între 0 și 1.
# Formula normalizării: (x - min) / (max - min)

df["salariu_normalizat"] = (df["salariu"] - df["salariu"].min()) / (df["salariu"].max() - df["salariu"].min())
print(df[["salariu","salariu_normalizat"]])
display_normalization()
# display_violinplot_compare(df)

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru normalizare (ex: grafic cu salariu/salariu normalizat)

# =======================================================
# 3️⃣ Standardization / Scalare (Z-score)
# =======================================================
# CERINȚĂ:
# Standardizează coloana 'experiență_ani' astfel încât să aibă media 0 și deviația standard 1.
# Formula scalarii: (x - media) / deviația_standard

df["experienta_ani_std"] = (df["experienta_ani"] - df["experienta_ani"].mean()) / df["experienta_ani"].std()        # std = fct de standardizare
print(df[["experienta_ani", "experienta_ani_std"]])
print(df["experienta_ani"].mean())
display_standardization()

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru standardizare (ex: grafic cu salariu/salariu normalizat)

# =======================================================
# 4️⃣ Discretization (Binning)
# =======================================================
# CERINȚĂ:
# Împarte coloana 'vârstă' în 3 categorii: 'Tânăr', 'Matur', 'Senior'

labels = ["Tanar", "Matur", "Senior"]
bins = [0, 30, 40, 100]

df["categorie_varsta"] = pd.cut(df["varsta"], bins = bins, labels = labels)     # exista mai multe implementari, asta este una dintre ele
print(df[["varsta", "categorie_varsta"]])
print(df["categorie_varsta"].value_counts())        # se vede cate persoane sunt in  fiecare categorie

#Varianta 2
conditions = [
    (df["varsta"] < 30),
    (df["varsta"].between(30, 40)),
    (df["varsta"] > 40)
]
choices = ["Tanar", "Matur", "Senior"]

df["categorie_varsta_bin"] = np.select(conditions, choices, default="Necunoscut")
print(df[["varsta", "categorie_varsta_bin"]])

# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru discretizare
# ❗ TEMA: Gaseste alta modalitate pentru discretizare

display_discretization()

# =======================================================
# 5️⃣ Categorical Imputation (completare categorii lipsă)
# =======================================================
# CERINȚĂ:
# Simulează o valoare lipsă în coloana 'departament' și completează cu cel mai frecvent departament

df.loc[2, "departament"] = np.nan
# df.loc[3, "departament"] = np.nan
# df.loc[4, "departament"] = np.nan
display_barplots(df, categorical_imputation(df, "departament"), ("Original", "Cat Imputation"), "departament")


cmcdp = df["departament"].mode()[0]         # functia mode se aplica pe un array - returneaza un array cu cele mai frecvente elem in ordine
df["departament_f"] = df["departament"].fillna(cmcdp)
print(df[["departament", "departament_f"]])
print(df)

plt.figure(figsize=(10,6))
sns.histplot(df["departament"], bins=10, color="orange", label="Original", kde=True)
sns.histplot(df["departament_f"], bins=10, color="skyblue", label="Imputation", kde=True)

plt.title("Distribuția departamentelor inainte si dupa imputation")
plt.xlabel("Departament")
plt.ylabel("Frecvență")
plt.legend()
plt.grid(alpha=0.3)



# ❗ TEMA: Afiseaza pe un grafic ceva semnificativ pentru categ imputation

# =======================================================
# 6️⃣ Noise Removal (Outlier detection)
# =======================================================
# CERINȚĂ:
# Identifică eventualii outlier-i în coloana 'salariu' folosind deviația standard.
# Detectarea se face comparând fiecare valoare cu media ± factor * deviația standard.

media_salarii = df["salariu"].mean()
deviatia_std = df["salariu"].std()
factor = 1          # distanta fata de medie in variabila
limita_superioara = media_salarii + factor * deviatia_std   # fara paranteze
limita_inferioara = media_salarii - factor * deviatia_std
masca = (df["salariu"] >= limita_inferioara) & (df["salariu"] <= limita_superioara)
# 0 < x < 10
# limita_inferioara <= "salariu" <= limita_superioara
# print(masca)
df_clean = df[masca]
# print(df_clean)
# print(df)
display_violinplots(df, df_clean, ("Salariu", "Salariu - Noise Removal"), "salariu")
display_boxplots(df, df_clean, ("Salariu", "Salariu - Noise Removal"), "salariu")
# =======================================================
# 7️⃣ Limiting Outliers (Clipping)
# =======================================================
# CERINȚĂ:
# În loc să eliminăm outlier-ii, putem să îi limităm la anumite valori maxime/minime (percentile).

lower_limit = df["salariu"].quantile(0.05)
upper_limit = df["salariu"].quantile(0.95)

df["clipped_salary"] = df["salariu"].clip(lower = lower_limit, upper = upper_limit)

print("\n 7. Inainte si dupa clipping:\n", df[["salariu","clipped_salary"]])


# =======================================================
# 8️⃣ Grouping și Aggregation
# =======================================================
# CERINȚĂ:
# Grupăm datele după 'departament' și calculăm statistici agregate.

grupare = df.groupby("departament").agg(
    numar_angajati = ("id","count")
)

print("\n 8. Grouping și Aggregation :\n", grupare)

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

# =======================================================
# 1️⃣0️⃣ DataFrame final
# =======================================================
# Afișăm DataFrame-ul complet după toate transformările și exercițiile de preprocessing
# plt.tight_layout()
plt.show()