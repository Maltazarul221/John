import pandas as pd

# Crearea DataFrame-ului
df = pd.read_csv("products-10000.csv")

print("#1 DataFrame complet:\n")
print(df)

# 2. Displaying data
print("#2 Ultimele 5 rânduri (tail):\n")
print(df.tail(5))
print("#2 Primele 5 rânduri (head):\n")
print(df.head(5))
print("#2 5 rânduri aleatoare (sample):\n")
print(df.sample(3))

# 3. Accessing columns
print("#3 Coloana 'produs' ca Series:")
print(df['produs'])
print("#3 Coloana 'produs' ca DataFrame:\n")
print()
print("#3 Coloanele 'produs' și 'categorie':\n")
print()

# 4. DataFrame metadata
print("#4 Indici (Indexes):")
print()
print("#4 Numele coloanelor (Column names):")
print()
print("#4 Informații despre dataset:")
print()

# 5. Subsetting rows/columns with .loc and .iloc
print("#5 Toate rândurile, coloana 'preț_unitar':\n")
print()
print("#5 Primele 6 rânduri, coloana 'preț_unitar':\n")
print()
print("#5 Toate rândurile, coloanele 'stoc' și 'preț_unitar':\n")
print()
print("#5 Rândul 3, coloanele 'stoc' și 'preț_unitar':\n")
print()

# 6. Filtering data with logical conditions
print("#6 Produsele cu stoc > 20:\n")
print()
print("#6 Produsele cu id între 3 și 7:\n")
print()
