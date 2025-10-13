import pandas as pd

# ----------------------------
# 1. Structuri principale
# ----------------------------
# Series – vector unidimensional etichetat
s = pd.Series([10, 20, 30])

# DataFrame – tabel bidimensional cu etichete
df = pd.DataFrame({
    'A': [1,2,3],
    'B': [4,5,6]
})

# Index – etichete pentru rânduri sau coloane
index = df.index
columns = df.columns

# ----------------------------
# 2. Citirea și scrierea datelor
# ----------------------------
df_csv = pd.read_csv('file.csv')       # CSV
df_tsv = pd.read_csv('file.tsv', sep='\t')  # TSV
df.to_parquet('file.parquet')
df_parquet = pd.read_parquet('file.parquet')
df.to_hdf('file.h5', key='df', format='table')
df_hdf = pd.read_hdf('file.h5', 'df')

# ----------------------------
# 3. Vizualizarea datelor
# ----------------------------
df.head()       # primele 5 rânduri
df.tail()       # ultimele 5 rânduri
df.sample(3)    # 3 rânduri aleatorii
df['A']         # accesare coloană (Series)
df[['A','B']]   # mai multe coloane (DataFrame)
df.info()       # tipuri și valori lipsă
df.index        # indexuri
df.columns      # nume coloane

# ----------------------------
# 4. Selectarea rândurilor / coloanelor
# ----------------------------
# .loc – după etichete
df.loc[0, 'A']          # un element
df.loc[:, ['A','B']]    # coloane multiple
df.loc[0:2, 'A']        # rânduri 0-2, col A

# .iloc – după poziții
df.iloc[0, 1]           # element rând 0, col 1
df.iloc[0:2, 0:2]       # primele 2 rânduri, primele 2 coloane

# ----------------------------
# 5. Filtrare condițională
# ----------------------------
df[df['A'] > 2]                   # rânduri cu A>2
df[(df['A']>2) & (df['B']<5)]     # combinații de condiții

# ----------------------------
# 6. Operații pe date
# ----------------------------
# apply(func) – aplică funcție pe fiecare element / coloană
df['A_squared'] = df['A'].apply(lambda x: x**2)

# groupby('col') – grupare pentru agregare
df.groupby('A').sum()
df.groupby('A')['B'].mean()

# ----------------------------
# 7. Ștergere date
# ----------------------------
df.drop(columns='B', inplace=False)  # șterge coloană
df.dropna()                          # șterge rânduri cu NaN

# ----------------------------
# 8. Sortare și resetare index
# ----------------------------
df_sorted = df.sort_values('A', ascending=False)
df_reset = df_sorted.reset_index(drop=True)

# ----------------------------
# 9. Merging / concatenation
# ----------------------------
df1 = df.head(2)
df2 = df.tail(2)

# Vertical (rânduri)
df_vertical = pd.concat([df1, df2], axis=0)

# Orizontal (coloane)
df_horizontal = pd.concat([df1, df2], axis=1)

# SQL-style join
df_merged = df1.merge(df2, left_index=True, right_index=True, how='inner')
