import pandas as pd

# Crearea DataFrame-ului

# 1. Reading the CSV file
df = pd.read_csv("products-10000.csv", encoding="utf-8-sig")

print("#1 DataFrame complet:\n")
print(df)
print()

# 2. Displaying data
print("#2 Ultimele 5 rânduri (tail):\n")
print(df.tail(5))
print()

print("#2 Primele 5 rânduri (head):\n")
print(df.head(5))
print()

print("#2 5 rânduri aleatoare (sample):\n")
print(df.sample(3))
print()

# 3. Accessing columns
print("#3 Coloana 'produs' ca Series:")
print(df["produs"])
print()

print("#3 Coloana 'produs' ca DataFrame:\n")
print(df[["produs"]])
print()

print("#3 Coloanele 'produs' și 'categorie':\n")
print(df[["produs", "categorie"]])
print()

# 4. DataFrame metadata
print("#4 Indici (Indexes):")
print(df.index)
print()

print("#4 Numele coloanelor (Column names):")
print(df.columns)
print()

print("#4 Informații despre dataset:")
print(df.info())
print()

# 5. Subsetting rows/columns with .loc and .iloc
print("#5 Toate rândurile, coloana 'preț_unitar':\n")
print(df.loc[:, "preț_unitar"])
print()

print("#5 Primele 6 rânduri, coloana 'preț_unitar':\n")
print(df.loc[:5, "preț_unitar"])
print()

print("#5 Toate rândurile, coloanele 'stoc' și 'preț_unitar':\n")
print(df.loc[:, ["stoc", "preț_unitar"]])
print()

print("#5 Rândul 3, coloanele 'stoc' și 'preț_unitar':\n")
print(df.loc[2, ["stoc", "preț_unitar"]])
print()

# 6. Filtering data with logical conditions
print("#6 Produsele cu stoc > 20:\n")
print(df[df["stoc"] > 20])
print()

print("#6 Produsele cu id între 3 și 7:\n")
print(df[(df["id"] >= 3) & (df["id"] <= 7)])
print()
