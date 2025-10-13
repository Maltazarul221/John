import pandas as pd

# Crearea DataFrame-ului
import pandas as pd

# 1. Reading the CSV file
df = pd.read_csv("angajati.csv")

print("#1 DataFrame:\n")
print(df)
print()

# 2. Displaying data
print("#2 First 5 rows (head):\n")
print(df.head())
print()

print("#2 Last 5 rows (tail):\n")
print(df.tail())
print()

print("#2 Random 5 rows (sample):\n")
print(df.sample(5))
print()

# 3. Accessing columns
print("#3 Single column as Series:")
print(df["salariu"])  # Example column
print()

print("#3 Single column as DataFrame:\n")
print(df[["salariu"]])
print()

print("#3 Multiple columns:\n")
print(df[["vârstă", "salariu"]])
print()

# 4. DataFrame metadata
print("#4 Indexes:")
print(df.index)
print()

print("#4 Column names:")
print(df.columns)
print()

print("#4 Info about dataset:")
print(df.info())
print()

# 5. Subsetting rows/columns with .loc and .iloc
print("#5 All rows, column 'salariu':\n")
print(df.loc[:, "salariu"])
print()

print("#5 First 6 rows, column 'salariu':\n")
print(df.loc[:5, "salariu"])
print()

print("#5 All rows, columns 'vârstă' and 'salariu':\n")
print(df.loc[:, ["vârstă", "salariu"]])
print()

print("#5 Row 3, columns 'vârstă' and 'salariu':\n")
print(df.loc[2, ["vârstă", "salariu"]])
print()

# 6. Filtering data with logical conditions
print("#6 Rows where salariu > 6000:\n")
print(df[df["salariu"] > 6000])
print()

print("#6 Rows where 3 <= id <= 7:\n")
print(df[(df["id"] >= 3) & (df["id"] <= 7)])
print()

