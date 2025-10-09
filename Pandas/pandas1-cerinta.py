import pandas as pd

# Crearea DataFrame-ului
df = pd.read_csv("angajati.csv")

print("#1 DataFrame:\n")
print()

# 2. Displaying data
print("#2 First 5 rows (head):\n")
print()
print("#2 Last 5 rows (tail):\n")
print()
print("#2 Random 5 rows (sample):\n")
print()

# 3. Accessing columns
print("#3 Single column as Series:")
print()
print("#3 Single column as DataFrame:\n")
print()
print("#3 Multiple columns:\n")
print()

# 4. DataFrame metadata
print("#4 Indexes:")
print()
print("#4 Column names:")
print()
print("#4 Info about dataset:")
print()

# 5. Subsetting rows/columns with .loc and .iloc
print("#5 All rows, column 'salariu':\n")
print()
print("#5 First 6 rows, column 'salariu':\n")
print()
print("#5 All rows, columns 'vârstă' and 'salariu':\n")
print()
print("#5 Row 3, columns 'vârstă' and 'salariu':\n")
print()

# 6. Filtering data with logical conditions
print("#6 Rows where salariu > 6000:\n")
print()
print("#6 Rows where 3 <= id <= 7:\n")
print()
