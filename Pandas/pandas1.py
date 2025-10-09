import pandas as pd

# Crearea DataFrame-ului
# data = {
#     "id": range(1, 11),
#     "nume": ["Ana", "Bogdan", "Carmen", "Dan", "Elena", "Florin", "Gina", "Horațiu", "Ioana", "Jianu"],
#     "departament": ["HR", "IT", "IT", "Finance", "HR", "Marketing", "Finance", "IT", "Marketing", "Finance"],
#     "oraș": ["București", "Cluj", "Iași", "Timișoara", "Cluj", "București", "Iași", "Cluj", "Timișoara", "București"],
#     "salariu": [4500., 7200., 6800., 5200., 4700., 6000., 5100., 7500., 6400., 5600.],
#     "vârstă": [25, 31, 29, 45, 26, 34, 41, 38, 28, 36],
#     "experiență_ani": [2, 7, 5, 20, 3, 10, 18, 12, 6, 15]
# }

df = pd.read_csv("angajati.csv")
print("#1 DataFrame:\n", df, "\n")

# 2. Displaying data
print("#2 First 5 rows (head):\n", df.head(), "\n")
print("#2 Last 5 rows (tail):\n", df.tail(), "\n")
print("#2 Random 5 rows (sample):\n", df.sample(5), "\n")

# 3. Accessing columns
print("#3 Single column as Series:", df['salariu'], "\n")
print("#3 Single column as DataFrame:\n", df[['salariu']], "\n")
print("#3 Multiple columns:\n", df[['vârstă', 'salariu']], "\n")

# 4. DataFrame metadata
print("#4 Indexes:", df.index, "\n")
print("#4 Column names:", df.columns, "\n")
print("#4 Info about dataset:")
df.info()
print("\n")

# 5. Subsetting rows/columns with .loc and .iloc
print("#5 All rows, column 'salariu':\n", df.loc[:, 'salariu'], "\n")
print("#5 First 6 rows, column 'salariu':\n", df.loc[:5, 'salariu'], "\n")
print("#5 All rows, columns 'vârstă' and 'salariu':\n", df.loc[:, ['vârstă', 'salariu']], "\n")
print("#5 Row 3, columns 'vârstă' and 'salariu':\n", df.loc[3, ['vârstă', 'salariu']], "\n")

# 6. Filtering data with logical conditions
print("#6 Rows where salariu > 6000:\n", df[df["salariu"] > 6000], "\n")
print("#6 Rows where 3 <= id <= 7:\n", df[(df["id"] >= 3) & (df["id"] <= 7)], "\n")
