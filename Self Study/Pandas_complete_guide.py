"""
================================================================================
                    GHID COMPLET PANDAS - Tutorial Detaliat
================================================================================

Pandas este biblioteca Python esențială pentru analiza și manipularea datelor.
Oferă structuri de date puternice (Series și DataFrame) și funcții pentru
lucrul cu date tabulare și time series.

Cuprins:
1. Instalare și Import
2. Structuri de Date: Series
3. Structuri de Date: DataFrame
4. Citirea și Scrierea Datelor
5. Vizualizarea Datelor
6. Selecția Datelor
7. Filtrarea și Interogarea
8. Modificarea Datelor
9. Operații cu Valori Lipsă
10. Operații de Agregare și Grupare
11. Merge, Join și Concatenare
12. Pivot Tables și Cross-tabulation
13. Time Series și Date
14. Funcții Apply și Map
15. Funcții String
16. Categorical Data
17. MultiIndex și Hierarchical Indexing
18. Optimizare și Performanță
19. Vizualizare cu Pandas
20. Best Practices și Cazuri Practice
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ============================================================================
# 1. INSTALARE ȘI IMPORT
# ============================================================================

"""
Instalare:
    pip install pandas
    pip install openpyxl  # Pentru Excel
    pip install xlrd      # Pentru Excel vechi

Import standard:
    import pandas as pd
    import numpy as np
"""

print("Versiunea Pandas:", pd.__version__)
print("Versiunea NumPy:", np.__version__)
print("\n" + "="*80 + "\n")

# ============================================================================
# 2. STRUCTURI DE DATE: SERIES
# ============================================================================

print("2. PANDAS SERIES\n")

"""
Series este un array unidimensional etichetat care poate conține orice tip de date.
Similar cu o coloană într-un Excel sau o listă Python cu index.
"""

# 2.1 Crearea unei Series
s1 = pd.Series([1, 2, 3, 4, 5])
print("Series simplă:")
print(s1)

# Series cu index custom
s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("\nSeries cu index custom:")
print(s2)

# Series din dicționar
data_dict = {'Ana': 25, 'Ion': 30, 'Maria': 22, 'Radu': 35}
s3 = pd.Series(data_dict)
print("\nSeries din dicționar:")
print(s3)

# Series cu un scalar
s4 = pd.Series(5, index=['a', 'b', 'c', 'd'])
print("\nSeries cu scalar:")
print(s4)

# 2.2 Proprietățile Series
print("\nProprietăți Series:")
print("Valori (values):", s3.values)
print("Index:", s3.index)
print("Tip date (dtype):", s3.dtype)
print("Shape:", s3.shape)
print("Size:", s3.size)
print("Nume:", s3.name if s3.name else "None")

# Setare nume
s3.name = "Vârste"
s3.index.name = "Persoane"
print("\nSeries cu nume:")
print(s3)

# 2.3 Accesare elemente
print("\nAccesare elemente:")
print("Element cu index 'Ana':", s3['Ana'])
print("Element cu poziție 0:", s3.iloc[0])
print("Elemente multiple:", s3[['Ana', 'Maria']])
print("Slicing:", s3['Ana':'Maria'])  # Include ultimul!

# 2.4 Operații cu Series
s_num = pd.Series([1, 2, 3, 4, 5])
print("\nOperații matematice:")
print("Original:", s_num.values)
print("s + 10:", (s_num + 10).values)
print("s * 2:", (s_num * 2).values)
print("s ** 2:", (s_num ** 2).values)

# Operații între Series
s_a = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s_b = pd.Series([4, 5, 6], index=['a', 'b', 'd'])
print("\nAdunare Series (union index):")
print(s_a + s_b)  # NaN pentru index-uri care nu se potrivesc

# 2.5 Metode utile
print("\nMetode statistice:")
print("Media:", s3.mean())
print("Mediana:", s3.median())
print("Suma:", s3.sum())
print("Min/Max:", s3.min(), "/", s3.max())
print("Descriere:", s3.describe())

# 2.6 Filtrare
print("\nFiltrare Series:")
print("Vârste > 25:")
print(s3[s3 > 25])

print("\n" + "="*80 + "\n")

# ============================================================================
# 3. STRUCTURI DE DATE: DATAFRAME
# ============================================================================

print("3. PANDAS DATAFRAME\n")

"""
DataFrame este o structură de date 2D etichetată cu coloane de tipuri 
potențial diferite. Similar cu un spreadsheet sau o tabelă SQL.
"""

# 3.1 Crearea unui DataFrame

# Din dicționar de liste
data = {
    'Nume': ['Ana', 'Ion', 'Maria', 'Radu', 'Elena'],
    'Vârstă': [25, 30, 22, 35, 28],
    'Oraș': ['București', 'Cluj', 'Iași', 'Timișoara', 'București'],
    'Salariu': [3000, 4500, 2800, 5200, 3800]
}
df = pd.DataFrame(data)
print("DataFrame din dicționar:")
print(df)

# Din listă de dicționare
data_list = [
    {'Nume': 'Ana', 'Vârstă': 25, 'Oraș': 'București'},
    {'Nume': 'Ion', 'Vârstă': 30, 'Oraș': 'Cluj'},
    {'Nume': 'Maria', 'Vârstă': 22, 'Oraș': 'Iași'}
]
df2 = pd.DataFrame(data_list)
print("\nDataFrame din listă de dicționare:")
print(df2)

# Din NumPy array
arr = np.random.rand(5, 3)
df3 = pd.DataFrame(arr,
                   columns=['Col1', 'Col2', 'Col3'],
                   index=['Row1', 'Row2', 'Row3', 'Row4', 'Row5'])
print("\nDataFrame din NumPy array:")
print(df3)

# 3.2 Proprietăți DataFrame
print("\nProprietăți DataFrame:")
print("Shape:", df.shape)
print("Dimensiuni:", df.ndim)
print("Size (număr elemente):", df.size)
print("Tipuri de date:\n", df.dtypes)
print("\nColoane:", df.columns.tolist())
print("Index:", df.index.tolist())

# 3.3 Vizualizare date
print("\nPrimele rânduri:")
print(df.head(3))

print("\nUltimele rânduri:")
print(df.tail(2))

print("\nEșantion aleator:")
print(df.sample(3))

print("\nInformații generale:")
df.info()

print("\nStatistici descriptive:")
print(df.describe())

# Include și coloane non-numerice
print("\nDescrie toate coloanele:")
print(df.describe(include='all'))

# 3.4 Accesare coloane
print("\nAccesare coloane:")
print("\nO coloană (Series):")
print(df['Nume'])

print("\nMultiple coloane (DataFrame):")
print(df[['Nume', 'Vârstă']])

# Ca atribut (doar pentru nume simple)
print("\nCa atribut:")
print(df.Nume.values)

# 3.5 Adăugare coloane
df['Bonus'] = df['Salariu'] * 0.1
print("\nDataFrame cu coloană nouă (Bonus):")
print(df)

# Coloană bazată pe condiție
df['Senior'] = df['Vârstă'] >= 30
print("\nDataFrame cu coloană condiționată:")
print(df)

# 3.6 Ștergere coloane
df_copy = df.copy()
df_copy = df_copy.drop('Senior', axis=1)
print("\nDataFrame după ștergere coloană:")
print(df_copy.head())

# Ștergere multiple coloane
df_copy2 = df.drop(['Bonus', 'Senior'], axis=1)
print("\nDataFrame după ștergere multiple coloane:")
print(df_copy2.head())

print("\n" + "="*80 + "\n")

# ============================================================================
# 4. CITIREA ȘI SCRIEREA DATELOR
# ============================================================================

print("4. CITIREA ȘI SCRIEREA DATELOR\n")

# 4.1 CSV
print("Operații CSV:")
# Creare exemplu CSV
df_csv = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd'],
    'C': [1.1, 2.2, 3.3, 4.4]
})

# Salvare
# df_csv.to_csv('date.csv', index=False)
# print("Salvat în date.csv")

# Citire
# df_citit = pd.read_csv('date.csv')
# print(df_citit)

print("Citire CSV - opțiuni comune:")
print("""
# Citire de bază
df = pd.read_csv('fisier.csv')

# Cu separator custom
df = pd.read_csv('fisier.txt', sep='\\t')

# Prima coloană ca index
df = pd.read_csv('fisier.csv', index_col=0)

# Selectare coloane specifice
df = pd.read_csv('fisier.csv', usecols=['Col1', 'Col2'])

# Specificare tipuri de date
df = pd.read_csv('fisier.csv', dtype={'Col1': str, 'Col2': int})

# Tratare valori lipsă
df = pd.read_csv('fisier.csv', na_values=['NA', 'N/A', ''])

# Skip rânduri
df = pd.read_csv('fisier.csv', skiprows=5)

# Citire parțială
df = pd.read_csv('fisier.csv', nrows=1000)

# Parse date
df = pd.read_csv('fisier.csv', parse_dates=['Data'])
""")

# 4.2 Excel
print("\nOperații Excel:")
print("""
# Citire Excel
df = pd.read_excel('fisier.xlsx', sheet_name='Sheet1')

# Citire multiple sheet-uri
dict_sheets = pd.read_excel('fisier.xlsx', sheet_name=None)

# Salvare Excel
df.to_excel('output.xlsx', sheet_name='Date', index=False)

# Multiple sheet-uri
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
""")

# 4.3 JSON
print("\nOperații JSON:")
print("""
# Citire JSON
df = pd.read_json('date.json')

# Salvare JSON
df.to_json('output.json', orient='records', indent=2)

# Orientări: 'split', 'records', 'index', 'columns', 'values'
""")

# 4.4 SQL
print("\nOperații SQL:")
print("""
import sqlite3

# Conectare
conn = sqlite3.connect('database.db')

# Citire din SQL
df = pd.read_sql('SELECT * FROM tabel', conn)
df = pd.read_sql_query('SELECT * FROM tabel WHERE col > 5', conn)
df = pd.read_sql_table('tabel', conn)

# Scriere în SQL
df.to_sql('tabel_nou', conn, if_exists='replace', index=False)
# if_exists: 'fail', 'replace', 'append'

conn.close()
""")

# 4.5 Alte formate
print("\nAlte formate:")
print("""
# HTML
df = pd.read_html('page.html')[0]  # Prima tabelă
df.to_html('output.html')

# Clipboard
df = pd.read_clipboard()
df.to_clipboard()

# Pickle (format Python)
df.to_pickle('data.pkl')
df = pd.read_pickle('data.pkl')

# Parquet (comprimat, rapid)
df.to_parquet('data.parquet')
df = pd.read_parquet('data.parquet')

# HDF5 (date mari)
df.to_hdf('data.h5', key='df', mode='w')
df = pd.read_hdf('data.h5', 'df')
""")

print("\n" + "="*80 + "\n")

# ============================================================================
# 5. VIZUALIZAREA DATELOR
# ============================================================================

print("5. VIZUALIZAREA DATELOR\n")

# Creăm un DataFrame pentru exemple
df_viz = pd.DataFrame({
    'Nume': ['Ana', 'Ion', 'Maria', 'Radu', 'Elena', 'Mihai', 'Laura'],
    'Vârstă': [25, 30, 22, 35, 28, 32, 27],
    'Salariu': [3000, 4500, 2800, 5200, 3800, 4200, 3500],
    'Departament': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'Sales'],
    'Experiență': [2, 5, 1, 8, 4, 6, 3]
})

print("Date pentru vizualizare:")
print(df_viz)

print("\nPrimele/Ultimele rânduri:")
print("head():\n", df_viz.head(3))
print("\ntail():\n", df_viz.tail(2))

print("\nInformații despre DataFrame:")
print("\ninfo():")
df_viz.info()

print("\ndescribe() - statistici numerice:")
print(df_viz.describe())

print("\ndescribe(include='all') - toate coloanele:")
print(df_viz.describe(include='all'))

print("\nValoare counts:")
print(df_viz['Departament'].value_counts())

print("\nValori unice:")
print("Departamente unice:", df_viz['Departament'].unique())
print("Număr departamente:", df_viz['Departament'].nunique())

print("\n" + "="*80 + "\n")

# ============================================================================
# 6. SELECȚIA DATELOR
# ============================================================================

print("6. SELECȚIA DATELOR\n")

df_sel = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['a', 'b', 'c', 'd', 'e'],
    'D': [1.1, 2.2, 3.3, 4.4, 5.5]
}, index=['row1', 'row2', 'row3', 'row4', 'row5'])

print("DataFrame pentru selecție:")
print(df_sel)

# 6.1 Selecție coloane
print("\nSelectare coloane:")
print("\nO coloană (Series):")
print(df_sel['A'])

print("\nMultiple coloane:")
print(df_sel[['A', 'C']])

# 6.2 loc - selecție bazată pe etichete
print("\nloc - selecție pe etichete:")
print("\nUn element:")
print(df_sel.loc['row1', 'A'])

print("\nO linie:")
print(df_sel.loc['row1'])

print("\nMultiple linii:")
print(df_sel.loc[['row1', 'row3']])

print("\nSlicing:")
print(df_sel.loc['row1':'row3'])  # Include ultimul!

print("\nLinii și coloane:")
print(df_sel.loc['row1':'row3', ['A', 'C']])

print("\nToate liniile, coloane specifice:")
print(df_sel.loc[:, ['A', 'B']])

# 6.3 iloc - selecție bazată pe poziție
print("\niloc - selecție pe poziție:")
print("\nUn element:")
print(df_sel.iloc[0, 0])

print("\nO linie:")
print(df_sel.iloc[0])

print("\nMultiple linii:")
print(df_sel.iloc[[0, 2]])

print("\nSlicing:")
print(df_sel.iloc[0:3])  # NU include ultimul!

print("\nLinii și coloane:")
print(df_sel.iloc[0:3, [0, 2]])

print("\nPas în slicing:")
print(df_sel.iloc[::2])  # Fiecare a doua linie

# 6.4 at și iat - acces rapid la scalar
print("\nat și iat - acces scalar rapid:")
print("at:", df_sel.at['row1', 'A'])
print("iat:", df_sel.iat[0, 0])

# 6.5 Selecție cu condiții booleene
print("\nSelecție booleană:")
print("\nA > 2:")
print(df_sel[df_sel['A'] > 2])

print("\nMultiple condiții (AND):")
print(df_sel[(df_sel['A'] > 2) & (df_sel['B'] < 50)])

print("\nMultiple condiții (OR):")
print(df_sel[(df_sel['A'] < 2) | (df_sel['A'] > 4)])

print("\nNegare:")
print(df_sel[~(df_sel['A'] > 3)])

# 6.6 isin și between
print("\nisin - element în listă:")
print(df_sel[df_sel['C'].isin(['a', 'c', 'e'])])

print("\nbetween - între valori:")
print(df_sel[df_sel['A'].between(2, 4)])

# 6.7 query - sintaxă SQL-like
print("\nquery - sintaxă SQL:")
print(df_sel.query('A > 2 and B < 50'))
print(df_sel.query('C in ["a", "c"]'))

print("\n" + "="*80 + "\n")

# ============================================================================
# 7. FILTRAREA ȘI INTEROGAREA
# ============================================================================

print("7. FILTRAREA ȘI INTEROGAREA\n")

df_filter = pd.DataFrame({
    'Produs': ['Laptop', 'Mouse', 'Tastatură', 'Monitor', 'Webcam', 'Boxe'],
    'Preț': [3000, 50, 150, 1200, 200, 300],
    'Categorie': ['IT', 'Accesorii', 'Accesorii', 'IT', 'Accesorii', 'Accesorii'],
    'Stoc': [15, 100, 50, 25, 30, 40],
    'Rating': [4.5, 4.0, 4.2, 4.7, 3.8, 4.1]
})

print("Date pentru filtrare:")
print(df_filter)

# 7.1 Filtrare simplă
print("\nProduse IT:")
print(df_filter[df_filter['Categorie'] == 'IT'])

print("\nProduse peste 200 lei:")
print(df_filter[df_filter['Preț'] > 200])

# 7.2 Filtrare complexă
print("\nProduse IT sub 2000 lei:")
print(df_filter[(df_filter['Categorie'] == 'IT') & (df_filter['Preț'] < 2000)])

print("\nProduse cu rating > 4 sau stoc < 30:")
print(df_filter[(df_filter['Rating'] > 4) | (df_filter['Stoc'] < 30)])

# 7.3 Filtrare cu string methods
print("\nProduse care conțin 'o':")
print(df_filter[df_filter['Produs'].str.contains('o', case=False)])

print("\nProduse care încep cu 'M':")
print(df_filter[df_filter['Produs'].str.startswith('M')])

# 7.4 Filtrare după index
print("\nProduse cu index par:")
print(df_filter[df_filter.index % 2 == 0])

# 7.5 nlargest și nsmallest
print("\nTop 3 cele mai scumpe:")
print(df_filter.nlargest(3, 'Preț'))

print("\nTop 3 cele mai ieftine:")
print(df_filter.nsmallest(3, 'Preț'))

# 7.6 where și mask
print("\nWhere - păstrează shape, înlocuiește cu NaN:")
print(df_filter.where(df_filter['Preț'] > 200))

print("\nMask - opusul lui where:")
print(df_filter.mask(df_filter['Preț'] > 200))

print("\n" + "="*80 + "\n")

# ============================================================================
# 8. MODIFICAREA DATELOR
# ============================================================================

print("8. MODIFICAREA DATELOR\n")

df_mod = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['a', 'b', 'c', 'd', 'e']
})

print("DataFrame original:")
print(df_mod)

# 8.1 Modificare valori
df_mod.loc[0, 'A'] = 100
print("\nDupă modificare element [0, 'A']:")
print(df_mod)

# Modificare coloană întreagă
df_mod_copy = df_mod.copy()
df_mod_copy['B'] = df_mod_copy['B'] * 2
print("\nColoană B dublată:")
print(df_mod_copy)

# Modificare bazată pe condiție
df_mod_copy2 = df_mod.copy()
df_mod_copy2.loc[df_mod_copy2['A'] > 50, 'A'] = 0
print("\nA > 50 devine 0:")
print(df_mod_copy2)

# 8.2 Adăugare coloane
df_mod_copy = df_mod.copy()
df_mod_copy['D'] = df_mod_copy['A'] + df_mod_copy['B']
print("\nAdăugare coloană D = A + B:")
print(df_mod_copy)

# Coloană condiționată
df_mod_copy['E'] = df_mod_copy['A'].apply(lambda x: 'mare' if x > 50 else 'mic')
print("\nColoană condiționată E:")
print(df_mod_copy)

# 8.3 Redenumire coloane
df_rename = df_mod.copy()
df_rename = df_rename.rename(columns={'A': 'Alpha', 'B': 'Beta'})
print("\nDupă redenumire coloane:")
print(df_rename)

# Redenumire toate coloanele
df_rename2 = df_mod.copy()
df_rename2.columns = ['Col1', 'Col2', 'Col3']
print("\nRedenumire toate coloanele:")
print(df_rename2)

# 8.4 Reindexare
df_reindex = df_mod.copy()
df_reindex.index = ['a', 'b', 'c', 'd', 'e']
print("\nDupă reindexare:")
print(df_reindex)

# Reset index
df_reset = df_reindex.reset_index(drop=False)
print("\nReset index (păstrează vechiul):")
print(df_reset)

# Set index
df_set = df_mod.copy()
df_set = df_set.set_index('C')
print("\nSet C ca index:")
print(df_set)

# 8.5 Sortare
df_sort = df_mod.copy()
print("\nSortare după coloană A (descrescător):")
print(df_sort.sort_values('A', ascending=False))

print("\nSortare după multiple coloane:")
df_multi = pd.DataFrame({
    'A': [3, 1, 2, 1, 3],
    'B': [5, 2, 4, 1, 3]
})
print(df_multi.sort_values(['A', 'B']))

# Sortare după index
print("\nSortare după index:")
df_idx = df_mod.copy()
df_idx.index = [4, 2, 0, 3, 1]
print(df_idx.sort_index())

# 8.6 Replace
df_replace = pd.DataFrame({
    'A': [1, 2, 3, -999, 5],
    'B': ['a', 'b', 'N/A', 'd', 'e']
})
print("\nDataFrame cu valori de înlocuit:")
print(df_replace)

df_replaced = df_replace.replace(-999, np.nan)
print("\nDupă replace -999 cu NaN:")
print(df_replaced)

df_replaced2 = df_replace.replace({'N/A': np.nan, -999: 0})
print("\nReplace multiple valori:")
print(df_replaced2)

print("\n" + "="*80 + "\n")

# ============================================================================
# 9. OPERAȚII CU VALORI LIPSĂ
# ============================================================================

print("9. OPERAȚII CU VALORI LIPSĂ (NaN)\n")

df_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, np.nan, 50],
    'C': ['a', 'b', 'c', 'd', 'e'],
    'D': [np.nan, np.nan, np.nan, np.nan, np.nan]
})

print("DataFrame cu valori lipsă:")
print(df_nan)

# 9.1 Detectare valori lipsă
print("\nisna() - detectare NaN:")
print(df_nan.isna())

print("\nNumăr NaN per coloană:")
print(df_nan.isna().sum())

print("\nProcent NaN per coloană:")
print(df_nan.isna().mean() * 100)

print("\nnotna() - detectare non-NaN:")
print(df_nan.notna().sum())

# 9.2 Eliminare valori lipsă
print("\ndropna() - elimină rânduri cu NaN:")
print(df_nan.dropna())

print("\ndropna(how='all') - elimină doar rânduri toate NaN:")
print(df_nan.dropna(how='all'))

print("\ndropna(axis=1) - elimină coloane cu NaN:")
print(df_nan.dropna(axis=1))

print("\ndropna(thresh=3) - păstrează rânduri cu minim 3 non-NaN:")
print(df_nan.dropna(thresh=3))

print("\ndropna(subset=['A', 'B']) - verifică doar coloane specifice:")
print(df_nan.dropna(subset=['A', 'B']))

# 9.3 Completare valori lipsă
print("\nfillna() - completare cu valoare:")
print(df_nan.fillna(0))

print("\nfillna() - completare diferită per coloană:")
print(df_nan.fillna({'A': 0, 'B': 999, 'D': -1}))

print("\nfillna(method='ffill') - forward fill:")
print(df_nan.fillna(method='ffill'))

print("\nfillna(method='bfill') - backward fill:")
print(df_nan.fillna(method='bfill'))

print("\nfillna cu medie:")
df_filled = df_nan.copy()
df_filled['A'] = df_filled['A'].fillna(df_filled['A'].mean())
df_filled['B'] = df_filled['B'].fillna(df_filled['B'].median())
print(df_filled)

# 9.4 Interpolation
df_interp = pd.DataFrame({
    'values': [1, np.nan, np.nan, 4, 5, np.nan, 7]
})
print("\nInterpolation:")
print("Original:")
print(df_interp)
print("\nLinear interpolation:")
print(df_interp.interpolate())

# 9.5 Replace valori specifice cu NaN
df_replace_nan = pd.DataFrame({
    'A': [1, -999, 3, -999, 5],
    'B': [10, 20, 0, 40, 50]
})
print("\nReplace valori specifice cu NaN:")
print("Original:")
print(df_replace_nan)
df_replace_nan = df_replace_nan.replace(-999, np.nan)
print("\nDupă replace -999 cu NaN:")
print(df_replace_nan)

print("\n" + "="*80 + "\n")

# ============================================================================
# 10. OPERAȚII DE AGREGARE ȘI GRUPARE
# ============================================================================

print("10. OPERAȚII DE AGREGARE ȘI GRUPARE\n")

df_agg = pd.DataFrame({
    'Departament': ['IT', 'IT', 'HR', 'HR', 'Sales', 'Sales', 'IT', 'HR'],
    'Angajat': ['Ana', 'Ion', 'Maria', 'Radu', 'Elena', 'Mihai', 'Laura', 'Bogdan'],
    'Salariu': [3000, 4500, 2800, 5200, 3800, 4200, 3500, 3200],
    'Experiență': [2, 5, 1, 8, 4, 6, 3, 2],
    'Proiecte': [5, 8, 3, 12, 6, 9, 7, 4]
})

print("Date pentru agregare:")
print(df_agg)

# 10.1 Funcții de agregare simple
print("\nFuncții de agregare:")
print("Medie salariu:", df_agg['Salariu'].mean())
print("Sumă salarii:", df_agg['Salariu'].sum())
print("Min/Max salariu:", df_agg['Salariu'].min(), "/", df_agg['Salariu'].max())
print("Mediană:", df_agg['Salariu'].median())
print("Std deviation:", df_agg['Salariu'].std())

# Aggregate pe multiple coloane
print("\nAggregate pe multiple coloane:")
print(df_agg[['Salariu', 'Experiență']].mean())

# 10.2 GroupBy - baza agregării în Pandas
print("\nGroupBy - grupare după departament:")
grouped = df_agg.groupby('Departament')

print("\nMedie salariu per departament:")
print(grouped['Salariu'].mean())

print("\nSumă salarii per departament:")
print(grouped['Salariu'].sum())

print("\nNumăr angajați per departament:")
print(grouped.size())

print("\nCount per departament:")
print(grouped['Angajat'].count())

# 10.3 Agregare multiplă
print("\nAgregare multiplă:")
print(grouped['Salariu'].agg(['mean', 'sum', 'min', 'max', 'count']))

# Funcții diferite pentru coloane diferite
print("\nFuncții diferite per coloană:")
result = grouped.agg({
    'Salariu': ['mean', 'sum'],
    'Experiență': ['mean', 'max'],
    'Proiecte': 'sum'
})
print(result)

# 10.4 Funcții custom
print("\nFuncție custom - range:")
print(grouped['Salariu'].agg(lambda x: x.max() - x.min()))

# 10.5 Transform - păstrează dimensiunea originală
print("\nTransform - salariu relativ la medie departament:")
df_transform = df_agg.copy()
df_transform['Salariu_relativ'] = grouped['Salariu'].transform(lambda x: x - x.mean())
print(df_transform)

# 10.6 Filter - filtrează grupuri
print("\nFilter - doar departamente cu >2 angajați:")
filtered = df_agg.groupby('Departament').filter(lambda x: len(x) > 2)
print(filtered)

# 10.7 Apply - operații complexe
print("\nApply - top 2 salarii per departament:")
top2 = df_agg.groupby('Departament').apply(lambda x: x.nlargest(2, 'Salariu'))
print(top2)

# 10.8 Multiple keys
df_multi_group = df_agg.copy()
df_multi_group['Nivel'] = ['Junior', 'Senior', 'Junior', 'Senior', 'Senior', 'Senior', 'Mid', 'Junior']

print("\nGroupBy cu multiple chei:")
print(df_multi_group.groupby(['Departament', 'Nivel'])['Salariu'].mean())

# 10.9 Iterare prin grupuri
print("\nIterare prin grupuri:")
for name, group in df_agg.groupby('Departament'):
    print(f"\n{name}:")
    print(group)

# 10.10 Named aggregation (Pandas 0.25+)
print("\nNamed aggregation:")
result = df_agg.groupby('Departament').agg(
    salariu_mediu=('Salariu', 'mean'),
    salariu_total=('Salariu', 'sum'),
    nr_angajati=('Angajat', 'count'),
    experienta_medie=('Experiență', 'mean')
)
print(result)

print("\n" + "="*80 + "\n")

# ============================================================================
# 11. MERGE, JOIN ȘI CONCATENARE
# ============================================================================

print("11. MERGE, JOIN ȘI CONCATENARE\n")

# Date pentru exemple
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Nume': ['Ana', 'Ion', 'Maria', 'Radu'],
    'Departament': ['IT', 'HR', 'Sales', 'IT']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 5, 6],
    'Salariu': [3000, 4500, 3800, 4200],
    'Oraș': ['București', 'Cluj', 'Iași', 'Timișoara']
})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# 11.1 Merge (similar SQL JOIN)
print("\nInner Join (implicit):")
merged_inner = pd.merge(df1, df2, on='ID')
print(merged_inner)

print("\nLeft Join:")
merged_left = pd.merge(df1, df2, on='ID', how='left')
print(merged_left)

print("\nRight Join:")
merged_right = pd.merge(df1, df2, on='ID', how='right')
print(merged_right)

print("\nOuter Join:")
merged_outer = pd.merge(df1, df2, on='ID', how='outer')
print(merged_outer)

# 11.2 Merge pe coloane cu nume diferite
df3 = pd.DataFrame({
    'AngajatID': [1, 2, 3],
    'Proiect': ['A', 'B', 'C']
})

print("\nMerge pe coloane diferite:")
merged_diff = pd.merge(df1, df3, left_on='ID', right_on='AngajatID')
print(merged_diff)

# 11.3 Merge pe multiple coloane
df4 = pd.DataFrame({
    'Nume': ['Ana', 'Ion', 'Maria'],
    'Departament': ['IT', 'HR', 'Sales'],
    'Bonus': [500, 600, 400]
})

print("\nMerge pe multiple coloane:")
merged_multi = pd.merge(df1, df4, on=['Nume', 'Departament'])
print(merged_multi)

# 11.4 Sufixe pentru coloane duplicate
df5 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Nume': ['Ana X', 'Ion Y', 'Maria Z'],
    'Scor': [85, 90, 88]
})

print("\nMerge cu sufixe:")
merged_suffix = pd.merge(df1, df5, on='ID', suffixes=('_angajat', '_evaluare'))
print(merged_suffix)

# 11.5 Join pe index
df_idx1 = df1.set_index('ID')
df_idx2 = df2.set_index('ID')

print("\nJoin pe index:")
joined = df_idx1.join(df_idx2, how='inner')
print(joined)

# 11.6 Concatenare
print("\nConcatenare verticală (axis=0):")
df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_b = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
concat_vert = pd.concat([df_a, df_b], ignore_index=True)
print(concat_vert)

print("\nConcatenare orizontală (axis=1):")
df_c = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})
concat_horiz = pd.concat([df_a, df_c], axis=1)
print(concat_horiz)

# 11.7 Append (depreciat, folosește concat)
print("\nAppend (folosește concat în loc):")
df_append = pd.DataFrame({'A': [7], 'B': [8]})
result_append = pd.concat([df_a, df_append], ignore_index=True)
print(result_append)

# 11.8 Merge cu validare
print("\nMerge cu validare one-to-one:")
try:
    validated = pd.merge(df1, df2, on='ID', validate='one_to_one')
    print("Merge valid!")
except Exception as e:
    print(f"Eroare: {e}")

print("\n" + "="*80 + "\n")

# ============================================================================
# 12. PIVOT TABLES ȘI CROSS-TABULATION
# ============================================================================

print("12. PIVOT TABLES ȘI CROSS-TABULATION\n")

# Date pentru pivot
df_pivot = pd.DataFrame({
    'Data': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-03', '2024-03'],
    'Categorie': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Produs': ['X', 'Y', 'X', 'Y', 'Z', 'Z'],
    'Vânzări': [100, 150, 120, 180, 90, 160],
    'Cantitate': [10, 15, 12, 18, 9, 16]
})

print("Date pentru pivot:")
print(df_pivot)

# 12.1 Pivot basic (necesită combinație unică index+columns)
# Pentru date cu duplicate, folosește pivot_table în schimb
print("\nPivot_table (recomandat pentru date cu posibile duplicate):")
pivot1 = df_pivot.pivot_table(
    values='Vânzări',
    index='Data',
    columns='Categorie',
    aggfunc='sum'
)
print(pivot1)

# 12.2 Pivot table cu agregare
print("\nPivot table cu sumă:")
pivot2 = df_pivot.pivot_table(
    values='Vânzări',
    index='Data',
    columns='Categorie',
    aggfunc='sum'
)
print(pivot2)

print("\nPivot table cu medie:")
pivot3 = df_pivot.pivot_table(
    values='Vânzări',
    index='Data',
    columns='Categorie',
    aggfunc='mean'
)
print(pivot3)

# 12.3 Multiple aggregation functions
print("\nPivot cu multiple funcții:")
pivot4 = df_pivot.pivot_table(
    values='Vânzări',
    index='Data',
    columns='Categorie',
    aggfunc=['sum', 'mean', 'count']
)
print(pivot4)

# 12.4 Multiple values
print("\nPivot cu multiple valori:")
pivot5 = df_pivot.pivot_table(
    values=['Vânzări', 'Cantitate'],
    index='Data',
    columns='Categorie',
    aggfunc='sum'
)
print(pivot5)

# 12.5 Margins (totals)
print("\nPivot cu totals (margins):")
pivot6 = df_pivot.pivot_table(
    values='Vânzări',
    index='Data',
    columns='Categorie',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)
print(pivot6)

# 12.6 Fill value
print("\nPivot cu fill_value:")
pivot7 = df_pivot.pivot_table(
    values='Vânzări',
    index='Data',
    columns='Produs',
    aggfunc='sum',
    fill_value=0
)
print(pivot7)

# 12.7 Cross-tabulation
df_cross = pd.DataFrame({
    'Gen': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'Fumător': ['Da', 'Nu', 'Da', 'Da', 'Nu', 'Nu', 'Da', 'Nu'],
    'Vârstă': ['<30', '<30', '30+', '30+', '<30', '30+', '30+', '<30']
})

print("\nDate pentru crosstab:")
print(df_cross)

print("\nCrosstab simplu:")
ct1 = pd.crosstab(df_cross['Gen'], df_cross['Fumător'])
print(ct1)

print("\nCrosstab cu proporții:")
ct2 = pd.crosstab(df_cross['Gen'], df_cross['Fumător'], normalize='index')
print(ct2)

print("\nCrosstab cu margins:")
ct3 = pd.crosstab(df_cross['Gen'], df_cross['Fumător'], margins=True)
print(ct3)

# 12.8 Melt (opus pivot - wide to long)
df_wide = pd.DataFrame({
    'ID': [1, 2, 3],
    'Ian': [100, 150, 200],
    'Feb': [110, 160, 210],
    'Mar': [120, 170, 220]
})

print("\nDataFrame wide:")
print(df_wide)

df_long = pd.melt(df_wide, id_vars=['ID'], var_name='Lună', value_name='Vânzări')
print("\nDataFrame long (după melt):")
print(df_long)

# 12.9 Stack și Unstack
df_stack = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['X', 'Y', 'Z'])

print("\nDataFrame pentru stack:")
print(df_stack)

stacked = df_stack.stack()
print("\nDupă stack:")
print(stacked)

unstacked = stacked.unstack()
print("\nDupă unstack:")
print(unstacked)

print("\n" + "="*80 + "\n")

# ============================================================================
# 13. TIME SERIES ȘI DATE
# ============================================================================

print("13. TIME SERIES ȘI DATE\n")

# 13.1 Creare date timestamps
print("Creare date:")
dates = pd.date_range('2024-01-01', periods=10, freq='D')
print("Date range (10 zile):")
print(dates)

dates_hours = pd.date_range('2024-01-01', periods=5, freq='H')
print("\nDate range (5 ore):")
print(dates_hours)

dates_months = pd.date_range('2024-01-01', periods=12, freq='M')
print("\nDate range (12 luni):")
print(dates_months)

# 13.2 Parsing strings to datetime
date_strings = ['2024-01-01', '2024-02-15', '2024-03-20']
parsed_dates = pd.to_datetime(date_strings)
print("\nParsing string to datetime:")
print(parsed_dates)

# Format custom
date_strings_custom = ['01/31/2024', '02/28/2024']
parsed_custom = pd.to_datetime(date_strings_custom, format='%m/%d/%Y')
print("\nParsing cu format custom:")
print(parsed_custom)

# 13.3 Time series DataFrame
df_ts = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=365, freq='D'),
    'valoare': np.random.randn(365).cumsum()
})
df_ts = df_ts.set_index('data')

print("\nTime series DataFrame:")
print(df_ts.head())

# 13.4 Selecție date
print("\nSelecție după dată:")
print(df_ts.loc['2024-01'])  # Ianuarie 2024

print("\nSelecție range de date:")
print(df_ts.loc['2024-01-01':'2024-01-05'])

# 13.5 Componente date
df_components = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=10, freq='D')
})

df_components['an'] = df_components['data'].dt.year
df_components['lună'] = df_components['data'].dt.month
df_components['zi'] = df_components['data'].dt.day
df_components['zi_săptămână'] = df_components['data'].dt.dayofweek
df_components['nume_zi'] = df_components['data'].dt.day_name()
df_components['trimestru'] = df_components['data'].dt.quarter

print("\nComponente date:")
print(df_components)

# 13.6 Operații cu date
date1 = pd.Timestamp('2024-01-01')
date2 = pd.Timestamp('2024-12-31')

print("\nOperații cu date:")
print(f"Diferență: {date2 - date1}")
print(f"Adăugare 10 zile: {date1 + pd.Timedelta(days=10)}")

# 13.7 Resampling
df_resample = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=100, freq='D'),
    'valoare': np.random.randn(100)
})
df_resample = df_resample.set_index('data')

print("\nResampling la nivel săptămânal (medie):")
print(df_resample.resample('W').mean().head())

print("\nResampling la nivel lunar (sumă):")
print(df_resample.resample('M').sum().head())

# 13.8 Rolling window
print("\nRolling mean (fereastră 7 zile):")
df_rolling = df_resample.copy()
df_rolling['rolling_mean'] = df_rolling['valoare'].rolling(window=7).mean()
print(df_rolling.head(10))

print("\nRolling cu multiple funcții:")
df_rolling_multi = df_resample['valoare'].rolling(window=7).agg(['mean', 'std', 'min', 'max'])
print(df_rolling_multi.head(10))

# 13.9 Shifting
print("\nShift (deplasare):")
df_shift = df_resample.head(5).copy()
df_shift['valoare_prev'] = df_shift['valoare'].shift(1)
df_shift['valoare_next'] = df_shift['valoare'].shift(-1)
print(df_shift)

# 13.10 Diferențe
print("\nDiferențe (diff):")
df_diff = df_resample.head(5).copy()
df_diff['diff'] = df_diff['valoare'].diff()
print(df_diff)

# 13.11 Time zones
print("\nTime zones:")
date_utc = pd.Timestamp('2024-01-01 12:00:00', tz='UTC')
print(f"UTC: {date_utc}")
date_romania = date_utc.tz_convert('Europe/Bucharest')
print(f"Romania: {date_romania}")

# 13.12 Business days
print("\nBusiness days:")
bdays = pd.bdate_range('2024-01-01', periods=10)
print(bdays)

# 13.13 Period
print("\nPeriods:")
periods = pd.period_range('2024-01', periods=12, freq='M')
print(periods)

# 13.14 Lag features pentru ML
df_lag = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=10, freq='D'),
    'valoare': range(10)
})
df_lag['lag_1'] = df_lag['valoare'].shift(1)
df_lag['lag_2'] = df_lag['valoare'].shift(2)
df_lag['lead_1'] = df_lag['valoare'].shift(-1)
print("\nLag features:")
print(df_lag)

print("\n" + "="*80 + "\n")

# ============================================================================
# 14. FUNCȚII APPLY ȘI MAP
# ============================================================================

print("14. FUNCȚII APPLY ȘI MAP\n")

df_apply = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['a', 'b', 'c', 'd', 'e']
})

print("DataFrame pentru apply:")
print(df_apply)

# 14.1 Apply pe Series
print("\nApply pe Series - dublare valori:")
result = df_apply['A'].apply(lambda x: x * 2)
print(result)

# Funcție cu nume
def patrat(x):
    return x ** 2

print("\nApply cu funcție definită:")
result2 = df_apply['A'].apply(patrat)
print(result2)

# 14.2 Apply pe DataFrame (pe coloane, axis=0)
print("\nApply pe coloane (axis=0) - sumă:")
result3 = df_apply[['A', 'B']].apply(sum, axis=0)
print(result3)

# 14.3 Apply pe rânduri (axis=1)
print("\nApply pe rânduri (axis=1) - sumă A+B:")
df_apply['suma'] = df_apply.apply(lambda row: row['A'] + row['B'], axis=1)
print(df_apply)

# 14.4 Apply cu multiple coloane
print("\nApply cu condiții complexe:")
def categorize(row):
    if row['A'] < 3:
        return 'Mic'
    elif row['A'] < 5:
        return 'Mediu'
    else:
        return 'Mare'

df_apply['categorie'] = df_apply.apply(categorize, axis=1)
print(df_apply)

# 14.5 Map - pentru Series
print("\nMap - mapare valori:")
mapping = {'a': 'Alpha', 'b': 'Beta', 'c': 'Gamma', 'd': 'Delta', 'e': 'Epsilon'}
df_apply['C_mapped'] = df_apply['C'].map(mapping)
print(df_apply)

# Map cu funcție
print("\nMap cu funcție:")
df_apply['B_squared'] = df_apply['B'].map(lambda x: x ** 2)
print(df_apply)

# 14.6 Applymap - pe tot DataFrame (depreciat, folosește map)
df_numeric = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

print("\nMap pe întreg DataFrame:")
result_applymap = df_numeric.map(lambda x: x * 10)
print(result_applymap)

# 14.7 Replace cu map
print("\nReplace folosind map:")
df_replace_map = pd.DataFrame({
    'status': ['active', 'inactive', 'active', 'pending']
})
status_map = {'active': 1, 'inactive': 0, 'pending': 2}
df_replace_map['status_code'] = df_replace_map['status'].map(status_map)
print(df_replace_map)

# 14.8 Apply cu progres (tqdm)
print("\nApply pentru operații complexe:")
print("Pentru date mari, consideră vectorizare sau numba/swifter")

# 14.9 Vectorizare vs Apply
print("\nComparație: Vectorizare vs Apply")
df_speed = pd.DataFrame({'A': range(10000)})

# Vectorizat (rapid)
import time
start = time.time()
result_vect = df_speed['A'] * 2
time_vect = time.time() - start

# Apply (mai lent)
start = time.time()
result_apply = df_speed['A'].apply(lambda x: x * 2)
time_apply = time.time() - start

print(f"Vectorizare: {time_vect:.4f}s")
print(f"Apply: {time_apply:.4f}s")
print(f"Apply este {time_apply/time_vect:.1f}x mai lent")

print("\n" + "="*80 + "\n")

# ============================================================================
# 15. FUNCȚII STRING
# ============================================================================

print("15. FUNCȚII STRING\n")

df_str = pd.DataFrame({
    'text': ['  Hello World  ', 'PYTHON pandas', 'Data Science', 'Machine Learning', 'AI ML'],
    'email': ['ana@example.com', 'ion@test.ro', 'maria@company.com', 'invalid', 'radu@site.org']
})

print("DataFrame pentru operații string:")
print(df_str)

# 15.1 Metode de bază
print("\nLower/Upper/Title:")
df_str['lower'] = df_str['text'].str.lower()
df_str['upper'] = df_str['text'].str.upper()
df_str['title'] = df_str['text'].str.title()
print(df_str[['text', 'lower', 'upper', 'title']])

# 15.2 Strip (eliminare spații)
print("\nStrip - eliminare spații:")
df_str['stripped'] = df_str['text'].str.strip()
print(df_str[['text', 'stripped']])

# 15.3 Contains (căutare substring)
print("\nContains - căutare substring:")
contains_data = df_str['text'].str.contains('Data', case=False)
print(contains_data)
print("\nRânduri cu 'Data':")
print(df_str[contains_data])

# 15.4 Startswith și Endswith
print("\nStartswith/Endswith:")
print("Începe cu 'Data':", df_str['text'].str.startswith('Data'))
print("Se termină cu 'ing':", df_str['text'].str.endswith('ing'))

# 15.5 Replace
print("\nReplace substring:")
df_str['replaced'] = df_str['text'].str.replace('Data', 'Big Data')
print(df_str[['text', 'replaced']])

# 15.6 Split
print("\nSplit:")
df_str['words'] = df_str['text'].str.split()
print(df_str[['text', 'words']])

# Split și expand
df_split = pd.DataFrame({
    'full_name': ['Ana Pop', 'Ion Ionescu', 'Maria Popescu']
})
df_split[['prenume', 'nume']] = df_split['full_name'].str.split(expand=True)
print("\nSplit expand:")
print(df_split)

# 15.7 Extract (regex)
print("\nExtract cu regex:")
df_str['domain'] = df_str['email'].str.extract(r'@(\w+)\.')
print(df_str[['email', 'domain']])

# 15.8 Length
print("\nLungime string:")
df_str['length'] = df_str['text'].str.len()
print(df_str[['text', 'length']])

# 15.9 Slice
print("\nSlice:")
df_str['first_5'] = df_str['text'].str[:5]
print(df_str[['text', 'first_5']])

# 15.10 Pad și center
print("\nPad și center:")
df_pad = pd.DataFrame({'text': ['a', 'bb', 'ccc']})
df_pad['padded'] = df_pad['text'].str.pad(width=5, side='both', fillchar='*')
df_pad['centered'] = df_pad['text'].str.center(5, fillchar='-')
print(df_pad)

# 15.11 Find și index
print("\nFind index:")
print(df_str['text'].str.find('a'))

# 15.12 Count occurrences
print("\nCount 'a':")
print(df_str['text'].str.lower().str.count('a'))

# 15.13 Validare email (regex)
print("\nValidare email:")
email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

# 15.14 Concatenare
print("\nConcatenare strings:")
df_concat = pd.DataFrame({
    'first': ['Ana', 'Ion'],
    'last': ['Pop', 'Ionescu']
})
df_concat['full'] = df_concat['first'] + ' ' + df_concat['last']
print(df_concat)

# Join
df_concat['joined'] = df_concat[['first', 'last']].agg(' '.join, axis=1)
print(df_concat)

print("\n" + "="*80 + "\n")

# ============================================================================
# 16. CATEGORICAL DATA
# ============================================================================

print("16. CATEGORICAL DATA\n")

# 16.1 Creare categorical
print("Creare categorical:")
sizes = pd.Series(['small', 'medium', 'large', 'small', 'medium', 'large', 'medium'])
sizes_cat = pd.Categorical(sizes)
print("Tip:", type(sizes_cat))
print("Categorii:", sizes_cat.categories)
print("Codes:", sizes_cat.codes)

# 16.2 Ordered categorical
print("\nOrdered categorical:")
sizes_ordered = pd.Categorical(
    sizes,
    categories=['small', 'medium', 'large'],
    ordered=True
)
print("Ordered:", sizes_ordered.ordered)
print("small < large:", sizes_ordered[0] < sizes_ordered[2])

# 16.3 Categorical în DataFrame
df_cat = pd.DataFrame({
    'size': pd.Categorical(['small', 'medium', 'large', 'small', 'medium'],
                          categories=['small', 'medium', 'large'],
                          ordered=True),
    'color': pd.Categorical(['red', 'blue', 'green', 'red', 'blue']),
    'price': [10, 20, 30, 15, 25]
})

print("\nDataFrame cu categorical:")
print(df_cat)
print("\nDtypes:")
print(df_cat.dtypes)

# 16.4 Conversie la categorical
df_convert = pd.DataFrame({
    'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A']
})
df_convert['grade'] = df_convert['grade'].astype('category')
print("\nDupă conversie la categorical:")
print(df_convert['grade'].dtype)

# 16.5 Avantaje categorical (memorie)
import sys
df_mem = pd.DataFrame({
    'category': ['cat1'] * 10000 + ['cat2'] * 10000 + ['cat3'] * 10000
})
mem_object = df_mem.memory_usage(deep=True)['category']

df_mem['category'] = df_mem['category'].astype('category')
mem_category = df_mem.memory_usage(deep=True)['category']

print(f"\nMemorie string: {mem_object / 1024:.2f} KB")
print(f"Memorie categorical: {mem_category / 1024:.2f} KB")
print(f"Economie: {(1 - mem_category/mem_object) * 100:.1f}%")

# 16.6 Operații cu categorical
print("\nValue counts pe categorical:")
print(df_cat['size'].value_counts())

print("\nDescribe pentru categorical:")
print(df_cat['color'].describe())

# 16.7 Adăugare/redenumire categorii
cat_series = pd.Series(pd.Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c']))
print("\nCategorii originale:", cat_series.cat.categories.tolist())

cat_series = cat_series.cat.add_categories(['d'])
print("După add_categories:", cat_series.cat.categories.tolist())

cat_series = cat_series.cat.rename_categories({'a': 'alpha', 'b': 'beta'})
print("După rename:", cat_series.cat.categories.tolist())

# 16.8 Eliminare categorii nefolosite
cat_with_unused = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c', 'd'])
print("\nCategorii cu unused:", cat_with_unused.categories.tolist())
cat_cleaned = cat_with_unused.remove_unused_categories()
print("După remove_unused:", cat_cleaned.categories.tolist())

# 16.9 Sortare categorical
print("\nSortare categorical ordered:")
df_sort_cat = df_cat.sort_values('size')
print(df_sort_cat)

print("\n" + "="*80 + "\n")

# ============================================================================
# 17. MULTIINDEX ȘI HIERARCHICAL INDEXING
# ============================================================================

print("17. MULTIINDEX ȘI HIERARCHICAL INDEXING\n")

# 17.1 Creare MultiIndex
arrays = [
    ['A', 'A', 'B', 'B', 'C', 'C'],
    ['one', 'two', 'one', 'two', 'one', 'two']
]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

df_multi = pd.DataFrame({
    'value1': [1, 2, 3, 4, 5, 6],
    'value2': [10, 20, 30, 40, 50, 60]
}, index=index)

print("DataFrame cu MultiIndex:")
print(df_multi)

# 17.2 Creare din tuples
tuples = [('A', 'one'), ('A', 'two'), ('B', 'one'), ('B', 'two')]
index_tuples = pd.MultiIndex.from_tuples(tuples, names=['letter', 'number'])
df_multi2 = pd.DataFrame({'value': [1, 2, 3, 4]}, index=index_tuples)
print("\nMultiIndex din tuples:")
print(df_multi2)

# 17.3 Creare din product (cartesian product)
index_product = pd.MultiIndex.from_product(
    [['A', 'B'], ['one', 'two', 'three']],
    names=['letter', 'number']
)
print("\nMultiIndex din product:")
print(index_product)

# 17.4 Selecție în MultiIndex
print("\nSelecție nivel 'A':")
print(df_multi.loc['A'])

print("\nSelecție specifică:")
print(df_multi.loc[('A', 'one')])

print("\nSelecție cu slice:")
print(df_multi.loc[('A', 'one'):('B', 'two')])

# 17.5 Cross-section (xs)
print("\nCross-section nivel 'one':")
print(df_multi.xs('one', level='second'))

# 17.6 Swaplevel
print("\nSwap levels:")
df_swapped = df_multi.swaplevel('first', 'second')
print(df_swapped)

# 17.7 Stack și Unstack cu MultiIndex
df_stacking = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['X', 'Y', 'Z'])

print("\nDataFrame original:")
print(df_stacking)

stacked = df_stacking.stack()
print("\nStacked (MultiIndex):")
print(stacked)

unstacked = stacked.unstack()
print("\nUnstacked:")
print(unstacked)

# 17.8 MultiIndex în coloane
df_multi_col = pd.DataFrame(
    np.random.rand(3, 6),
    columns=pd.MultiIndex.from_product([['A', 'B'], ['one', 'two', 'three']]),
    index=['X', 'Y', 'Z']
)
print("\nMultiIndex în coloane:")
print(df_multi_col)

print("\nSelectare coloană 'A':")
print(df_multi_col['A'])

# 17.9 Set și Reset MultiIndex
df_reset = df_multi.reset_index()
print("\nReset MultiIndex:")
print(df_reset)

df_set_multi = df_reset.set_index(['first', 'second'])
print("\nSet MultiIndex:")
print(df_set_multi)

# 17.10 Sortare MultiIndex
print("\nSortare MultiIndex:")
df_sorted = df_multi.sort_index()
print(df_sorted)

print("\nSortare descrescător nivel 'second':")
df_sorted2 = df_multi.sort_index(level='second', ascending=False)
print(df_sorted2)

# 17.11 GroupBy cu MultiIndex
df_group_multi = pd.DataFrame({
    'Region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250, 120, 180]
})

grouped_multi = df_group_multi.groupby(['Region', 'Product'])['Sales'].sum()
print("\nGroupBy rezultă MultiIndex:")
print(grouped_multi)

print("\n" + "="*80 + "\n")

# ============================================================================
# 18. OPTIMIZARE ȘI PERFORMANȚĂ
# ============================================================================

print("18. OPTIMIZARE ȘI PERFORMANȚĂ\n")

# 18.1 Alegerea tipului de date corect
print("Optimizare tipuri de date:")

df_optimize = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 10000),
    'float_col': np.random.rand(10000),
    'bool_col': np.random.choice([True, False], 10000),
    'cat_col': np.random.choice(['A', 'B', 'C', 'D'], 10000)
})

print("Memorie înainte de optimizare:")
mem_before = df_optimize.memory_usage(deep=True).sum() / 1024
print(f"{mem_before:.2f} KB")

# Optimizare int
df_optimize['int_col'] = df_optimize['int_col'].astype('int8')

# Optimizare float
df_optimize['float_col'] = df_optimize['float_col'].astype('float32')

# Optimizare categorical
df_optimize['cat_col'] = df_optimize['cat_col'].astype('category')

print("\nMemorie după optimizare:")
mem_after = df_optimize.memory_usage(deep=True).sum() / 1024
print(f"{mem_after:.2f} KB")
print(f"Economie: {(1 - mem_after/mem_before) * 100:.1f}%")

# 18.2 Vectorizare vs Iterare
print("\nVectorizare vs Iterare:")
df_speed = pd.DataFrame({'A': range(10000), 'B': range(10000)})

# Iterare (LENT)
start = time.time()
result_iter = []
for i in range(len(df_speed)):
    result_iter.append(df_speed.loc[i, 'A'] + df_speed.loc[i, 'B'])
time_iter = time.time() - start

# Apply (MAI RAPID)
start = time.time()
result_apply = df_speed.apply(lambda row: row['A'] + row['B'], axis=1)
time_apply = time.time() - start

# Vectorizare (CEL MAI RAPID)
start = time.time()
result_vect = df_speed['A'] + df_speed['B']
time_vect = time.time() - start

print(f"Iterare: {time_iter:.4f}s")
print(f"Apply: {time_apply:.4f}s")
print(f"Vectorizare: {time_vect:.4f}s")
print(f"Vectorizare este {time_iter/time_vect:.0f}x mai rapid decât iterare!")

# 18.3 Chunk processing pentru fișiere mari
print("\nChunk processing pentru CSV mari:")
chunk_example = """
# Citire în chunks
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Procesare chunk
    processed = chunk[chunk['column'] > 0]
    chunks.append(processed)

df = pd.concat(chunks, ignore_index=True)
"""
print(chunk_example)

# 18.4 Utilizare query pentru filtrare
df_query = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000)
})

start = time.time()
result_bool = df_query[(df_query['A'] > 0.5) & (df_query['B'] < 0.5)]
time_bool = time.time() - start

start = time.time()
result_query = df_query.query('A > 0.5 and B < 0.5')
time_query = time.time() - start

print(f"\nFiltrare boolean: {time_bool:.4f}s")
print(f"Filtrare query: {time_query:.4f}s")

# 18.5 Eval pentru expresii
df_eval = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000),
    'C': np.random.rand(100000)
})

start = time.time()
result_normal = df_eval['A'] + df_eval['B'] * df_eval['C']
time_normal = time.time() - start

start = time.time()
result_eval = df_eval.eval('A + B * C')
time_eval = time.time() - start

print(f"\nExpresie normală: {time_normal:.4f}s")
print(f"Expresie eval: {time_eval:.4f}s")

# 18.6 Copy vs View
print("\nCopy vs View:")
df_original = pd.DataFrame({'A': [1, 2, 3]})

# View (nu copiază date)
df_view = df_original[df_original['A'] > 1]  # Poate fi view sau copy

# Copy explicit
df_copy = df_original.copy()

print("Folosește .copy() când vrei să eviți SettingWithCopyWarning")

# 18.7 Inplace operations
df_inplace = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Fără inplace (creează copie)
df_new = df_inplace.drop('B', axis=1)

# Cu inplace (modifică original)
df_inplace_copy = df_inplace.copy()
df_inplace_copy.drop('B', axis=1, inplace=True)

print("\nInplace poate economisi memorie dar este adesea mai puțin eficient")
print("Preferă reatribuirea: df = df.drop('B', axis=1)")

# 18.8 Index optimization
print("\nIndex optimization:")
df_no_index = pd.DataFrame({
    'key': np.random.choice(['A', 'B', 'C'], 100000),
    'value': np.random.rand(100000)
})

df_with_index = df_no_index.set_index('key')

start = time.time()
result_no_idx = df_no_index[df_no_index['key'] == 'A']
time_no_idx = time.time() - start

start = time.time()
result_idx = df_with_index.loc['A']
time_idx = time.time() - start

print(f"Fără index: {time_no_idx:.4f}s")
print(f"Cu index: {time_idx:.4f}s")
print("Indexarea poate accelera selectarea!")

# 18.9 Sfaturi generale
print("\n" + "-"*80)
print("SFATURI PENTRU PERFORMANȚĂ:")
print("-"*80)
print("""
1. Folosește tipuri de date corespunzătoare (int8 vs int64, category)
2. Vectorizează operațiile - evită loop-uri și iterrows()
3. Folosește query() și eval() pentru operații complexe
4. Procesează fișiere mari în chunks
5. Folosește index pentru căutări rapide
6. Evită copii inutile - folosește view-uri când e posibil
7. Folosește categorical pentru coloane cu puține valori unice
8. Filtrează date devreme pentru a reduce volumul
9. Folosește numba sau Cython pentru operații complexe
10. Consideră Dask sau Polars pentru date foarte mari (>RAM)
""")

print("\n" + "="*80 + "\n")

# ============================================================================
# 19. VIZUALIZARE CU PANDAS
# ============================================================================

print("19. VIZUALIZARE CU PANDAS\n")

print("Pandas oferă integrare cu matplotlib pentru vizualizări rapide:")

df_plot = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=100, freq='D'),
    'valoare1': np.random.randn(100).cumsum(),
    'valoare2': np.random.randn(100).cumsum(),
    'categorie': np.random.choice(['A', 'B', 'C'], 100)
})

print("\nDate pentru vizualizare:")
print(df_plot.head())

print("\nTipuri de plot-uri disponibile:")
print("""
# Line plot
df_plot.plot(x='data', y='valoare1')
df_plot.plot(x='data', y=['valoare1', 'valoare2'])

# Bar plot
df_plot['categorie'].value_counts().plot(kind='bar')
df_plot.plot.bar(x='categorie', y='valoare1')

# Histogram
df_plot['valoare1'].plot(kind='hist', bins=20)
df_plot.plot.hist(alpha=0.5)

# Box plot
df_plot[['valoare1', 'valoare2']].plot(kind='box')

# Scatter plot
df_plot.plot.scatter(x='valoare1', y='valoare2')

# Area plot
df_plot.plot.area()

# Pie chart
df_plot['categorie'].value_counts().plot(kind='pie')

# Hexbin
df_plot.plot.hexbin(x='valoare1', y='valoare2', gridsize=15)

# KDE (density)
df_plot['valoare1'].plot(kind='kde')
""")

print("\nParametri comuni:")
print("""
- figsize=(width, height): dimensiune figură
- title='Titlu': titlu grafic
- xlabel, ylabel: etichete axe
- legend=True/False: legendă
- color: culoare
- alpha: transparență
- grid=True: grilă
- style: stil linie
- rot: rotație etichete
""")

print("\nExemplu complet:")
print("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
df_plot.plot(x='data', y=['valoare1', 'valoare2'], 
             ax=axes[0, 0], title='Time Series')

# Histogram
df_plot['valoare1'].plot(kind='hist', bins=20, 
                          ax=axes[0, 1], title='Histogramă')

# Box plot
df_plot[['valoare1', 'valoare2']].plot(kind='box', 
                                        ax=axes[1, 0], title='Box Plot')

# Bar plot
df_plot['categorie'].value_counts().plot(kind='bar', 
                                          ax=axes[1, 1], title='Categorii')

plt.tight_layout()
plt.show()
""")

print("\n" + "="*80 + "\n")

# ============================================================================
# 20. BEST PRACTICES ȘI CAZURI PRACTICE
# ============================================================================

print("20. BEST PRACTICES ȘI CAZURI PRACTICE\n")

print("BEST PRACTICES:")
print("-" * 80)
print("""
1. CITIREA DATELOR:
   - Specifică dtype-uri la citire pentru performanță
   - Folosește parse_dates pentru coloane de date
   - Citește în chunks pentru fișiere mari

2. CURĂȚAREA DATELOR:
   - Verifică și tratează valori lipsă consistent
   - Convertește tipuri de date corespunzător
   - Elimină duplicate
   - Validează date (range-uri, valori posibile)

3. MANIPULAREA DATELOR:
   - Folosește operații vectorizate
   - Evită iterrows() și itertuples() când e posibil
   - Folosește method chaining pentru cod mai clar
   - Set index pentru căutări frecvente

4. GRUPARE ȘI AGREGARE:
   - Folosește groupby eficient
   - Consideră transform vs apply vs agg
   - Named aggregation pentru claritate

5. PERFORMANȚĂ:
   - Alege tipurile de date potrivite
   - Folosește categorical pentru coloane cu puține valori
   - Evită copii inutile
   - Profită de index-uri

6. COD CURAT:
   - Method chaining pentru operații multiple
   - Nume descriptive pentru coloane
   - Comentează operații complexe
   - Verifică shape și dtypes frecvent
""")

print("\n" + "-"*80)
print("ERORI COMUNE:")
print("-"*80)
print("""
1. SettingWithCopyWarning:
   Problemă: df[df['A'] > 5]['B'] = 10
   Soluție: df.loc[df['A'] > 5, 'B'] = 10

2. Modificare DataFrame în iterație:
   Problemă: for idx, row in df.iterrows(): df.loc[idx] = ...
   Soluție: Folosește vectorizare sau apply

3. Confuzie între Series și DataFrame:
   Problemă: df['col'] vs df[['col']]
   Prima returnează Series, a doua DataFrame

4. Index nesetat după sortare:
   Problemă: df.sort_values('col')
   Soluție: df.sort_values('col').reset_index(drop=True)

5. Conversii implicite de tip:
   Problemă: Citire CSV fără dtype
   Soluție: Specifică dtype-uri explicit

6. Comparații cu NaN:
   Problemă: df['col'] == np.nan
   Soluție: df['col'].isna()

7. Folosire inplace când nu e necesar:
   Problemă: df.drop('col', inplace=True)
   Soluție: df = df.drop('col')  # Mai clar și adesea mai rapid
""")

print("\n" + "-"*80)
print("CAZURI PRACTICE:")
print("-"*80)

# Caz 1: Curățare date
print("\n1. CURĂȚARE DATE:")
df_dirty = pd.DataFrame({
    'nume': ['  Ana  ', 'ION', 'maria', 'RADU'],
    'vârstă': [25, -999, 30, 150],
    'salariu': ['3000', '4500.5', 'N/A', '5200']
})

print("Date murdare:")
print(df_dirty)

# Curățare
df_clean = df_dirty.copy()
df_clean['nume'] = df_clean['nume'].str.strip().str.title()
df_clean['vârstă'] = pd.to_numeric(df_clean['vârstă'], errors='coerce')
df_clean['vârstă'] = df_clean['vârstă'].apply(lambda x: x if 0 < x < 120 else np.nan)
df_clean['salariu'] = pd.to_numeric(df_clean['salariu'], errors='coerce')

print("\nDate curate:")
print(df_clean)

# Caz 2: Feature Engineering
print("\n2. FEATURE ENGINEERING:")
df_fe = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=10, freq='D'),
    'tranzacții': [5, 8, 3, 12, 6, 9, 4, 7, 11, 5],
    'valoare': [100, 200, 50, 300, 150, 250, 80, 180, 280, 120]
})

# Extragere features temporale
df_fe['zi_săptămână'] = df_fe['data'].dt.dayofweek
df_fe['este_weekend'] = df_fe['zi_săptămână'].isin([5, 6])
df_fe['săptămână_lună'] = df_fe['data'].dt.isocalendar().week % 4 + 1

# Features derivate
df_fe['valoare_medie'] = df_fe['valoare'] / df_fe['tranzacții']
df_fe['tranzacții_cumulative'] = df_fe['tranzacții'].cumsum()

# Lag features
df_fe['tranzacții_prev'] = df_fe['tranzacții'].shift(1)
df_fe['tranzacții_diff'] = df_fe['tranzacții'].diff()

# Rolling statistics
df_fe['tranzacții_ma3'] = df_fe['tranzacții'].rolling(window=3).mean()

print(df_fe)

# Caz 3: Transformare wide to long
print("\n3. TRANSFORMARE WIDE TO LONG:")
df_wide = pd.DataFrame({
    'ID': [1, 2, 3],
    'Nume': ['Ana', 'Ion', 'Maria'],
    'Q1_2024': [100, 150, 200],
    'Q2_2024': [110, 160, 210],
    'Q3_2024': [120, 170, 220]
})

print("Wide format:")
print(df_wide)

df_long = pd.melt(
    df_wide,
    id_vars=['ID', 'Nume'],
    var_name='Trimestru',
    value_name='Vânzări'
)

print("\nLong format:")
print(df_long)

# Caz 4: Deduplicare complexă
print("\n4. DEDUPLICARE COMPLEXĂ:")
df_dup = pd.DataFrame({
    'nume': ['Ana', 'Ana', 'Ion', 'Ion', 'Maria'],
    'data': pd.date_range('2024-01-01', periods=5, freq='D'),
    'valoare': [100, 150, 200, 200, 300]
})

print("Date cu duplicate:")
print(df_dup)

# Păstrează ultimul
df_dedup = df_dup.drop_duplicates(subset=['nume'], keep='last')
print("\nPăstrează ultimul per nume:")
print(df_dedup)

# Păstrează valoarea maximă
df_max = df_dup.loc[df_dup.groupby('nume')['valoare'].idxmax()]
print("\nPăstrează valoarea maximă per nume:")
print(df_max)

# Caz 5: Agregare complexă cu condiții
print("\n5. AGREGARE COMPLEXĂ:")
df_sales = pd.DataFrame({
    'Regiune': ['Nord', 'Sud', 'Nord', 'Sud', 'Nord', 'Sud'],
    'Produs': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Vânzări': [100, 150, 200, 180, 120, 220],
    'Cantitate': [10, 15, 20, 18, 12, 22]
})

result_complex = df_sales.groupby('Regiune').agg(
    vânzări_totale=('Vânzări', 'sum'),
    vânzări_medii=('Vânzări', 'mean'),
    cantitate_totală=('Cantitate', 'sum'),
    nr_produse=('Produs', 'nunique'),
    top_vânzare=('Vânzări', 'max')
).round(2)

print(result_complex)

print("\n" + "="*80 + "\n")

# ============================================================================
# CONCLUZIE ȘI RESURSE
# ============================================================================

print("CONCLUZIE ȘI RESURSE\n")
print("="*80)

print("""
DOCUMENTAȚIE ȘI RESURSE:
------------------------
- Pandas Documentation: https://pandas.pydata.org/docs/
- User Guide: https://pandas.pydata.org/docs/user_guide/index.html
- API Reference: https://pandas.pydata.org/docs/reference/index.html
- Pandas Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

CĂRȚI RECOMANDATE:
------------------
- "Python for Data Analysis" de Wes McKinney (creatorul Pandas)
- "Pandas Cookbook" de Matt Harrison
- "Effective Pandas" de Matt Harrison

BIBLIOTECI COMPLEMENTARE:
--------------------------
- NumPy: Operații numerice de bază
- Matplotlib/Seaborn: Vizualizare
- Scikit-learn: Machine learning
- Statsmodels: Analiză statistică
- Dask: Pandas paralel pentru date mari
- Polars: Alternativă rapidă la Pandas

PRACTICĂ:
---------
1. Kaggle datasets și competiții
2. Real World Datasets (data.gov, data.world)
3. Projece personale cu date reale
4. Contribuții open-source
5. Stack Overflow - răspunde întrebări

SFATURI FINALE:
---------------
- Pandas este esențial pentru data science în Python
- Practică regulat cu dataset-uri reale
- Citește documentația - este foarte bună
- Învață să gândești în termeni de operații vectorizate
- Explorează Pandas profiling și pandas-ta pentru productivitate
- Învață SQL - conceptele se traduc bine în Pandas
- Pentru date foarte mari, consideră Dask sau Polars

================================================================================
                    SUCCES ÎN ANALIZA DATELOR CU PANDAS!
================================================================================

Acest ghid acoperă toate aspectele fundamentale și avansate ale Pandas.
Practică exemplele, experimentează cu propriile date și construiește proiecte!

Autor: Ghid Complet Pandas
Data: 2025
Versiune: Completă și actualizată cu best practices
""")

print("="*80)
print("FINAL - Ghid Complet Pandas")
print("="*80)
df_str['valid_email'] = df_str['email'].str.match(email_pattern)
print(df_str[['email', 'valid_email']])

# 15.14 Concatenare
print("\nConcatenare strings:")
df_concat = pd.DataFrame({
    'first': ['Ana', 'Ion'],
    'last': ['Pop', 'Ionescu']
})
df_concat['full'] = df_concat['first'] + ' ' + df_concat['last']
print(df_concat)

# Join
df_concat['joined'] = df_concat[['first', 'last']].agg(' '.join, axis=1)
print(df_concat)

print("\n" + "="*80 + "\n")

# ============================================================================
# 16. CATEGORICAL DATA
# ============================================================================

print("16. CATEGORICAL DATA\n")

# 16.1 Creare categorical
print("Creare categorical:")
sizes = pd.Series(['small', 'medium', 'large', 'small', 'medium', 'large', 'medium'])
sizes_cat = pd.Categorical(sizes)
print("Tip:", type(sizes_cat))
print("Categorii:", sizes_cat.categories)
print("Codes:", sizes_cat.codes)

# 16.2 Ordered categorical
print("\nOrdered categorical:")
sizes_ordered = pd.Categorical(
    sizes,
    categories=['small', 'medium', 'large'],
    ordered=True
)
print("Ordered:", sizes_ordered.ordered)
print("small < large:", sizes_ordered[0] < sizes_ordered[2])

# 16.3 Categorical în DataFrame
df_cat = pd.DataFrame({
    'size': pd.Categorical(['small', 'medium', 'large', 'small', 'medium'],
                          categories=['small', 'medium', 'large'],
                          ordered=True),
    'color': pd.Categorical(['red', 'blue', 'green', 'red', 'blue']),
    'price': [10, 20, 30, 15, 25]
})

print("\nDataFrame cu categorical:")
print(df_cat)
print("\nDtypes:")
print(df_cat.dtypes)

# 16.4 Conversie la categorical
df_convert = pd.DataFrame({
    'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A']
})
df_convert['grade'] = df_convert['grade'].astype('category')
print("\nDupă conversie la categorical:")
print(df_convert['grade'].dtype)

# 16.5 Avantaje categorical (memorie)
import sys
df_mem = pd.DataFrame({
    'category': ['cat1'] * 10000 + ['cat2'] * 10000 + ['cat3'] * 10000
})
mem_object = df_mem.memory_usage(deep=True)['category']

df_mem['category'] = df_mem['category'].astype('category')
mem_category = df_mem.memory_usage(deep=True)['category']

print(f"\nMemorie string: {mem_object / 1024:.2f} KB")
print(f"Memorie categorical: {mem_category / 1024:.2f} KB")
print(f"Economie: {(1 - mem_category/mem_object) * 100:.1f}%")

# 16.6 Operații cu categorical
print("\nValue counts pe categorical:")
print(df_cat['size'].value_counts())

print("\nDescribe pentru categorical:")
print(df_cat['color'].describe())

# 16.7 Adăugare/redenumire categorii
cat_series = pd.Series(pd.Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c']))
print("\nCategorii originale:", cat_series.cat.categories.tolist())

cat_series = cat_series.cat.add_categories(['d'])
print("După add_categories:", cat_series.cat.categories.tolist())

cat_series = cat_series.cat.rename_categories({'a': 'alpha', 'b': 'beta'})
print("După rename:", cat_series.cat.categories.tolist())

# 16.8 Eliminare categorii nefolosite
cat_with_unused = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c', 'd'])
print("\nCategorii cu unused:", cat_with_unused.categories.tolist())
cat_cleaned = cat_with_unused.remove_unused_categories()
print("După remove_unused:", cat_cleaned.categories.tolist())

# 16.9 Sortare categorical
print("\nSortare categorical ordered:")
df_sort_cat = df_cat.sort_values('size')
print(df_sort_cat)

print("\n" + "="*80 + "\n")

# ============================================================================
# 17. MULTIINDEX ȘI HIERARCHICAL INDEXING
# ============================================================================

print("17. MULTIINDEX ȘI HIERARCHICAL INDEXING\n")

# 17.1 Creare MultiIndex
arrays = [
    ['A', 'A', 'B', 'B', 'C', 'C'],
    ['one', 'two', 'one', 'two', 'one', 'two']
]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

df_multi = pd.DataFrame({
    'value1': [1, 2, 3, 4, 5, 6],
    'value2': [10, 20, 30, 40, 50, 60]
}, index=index)

print("DataFrame cu MultiIndex:")
print(df_multi)

# 17.2 Creare din tuples
tuples = [('A', 'one'), ('A', 'two'), ('B', 'one'), ('B', 'two')]
index_tuples = pd.MultiIndex.from_tuples(tuples, names=['letter', 'number'])
df_multi2 = pd.DataFrame({'value': [1, 2, 3, 4]}, index=index_tuples)
print("\nMultiIndex din tuples:")
print(df_multi2)

# 17.3 Creare din product (cartesian product)
index_product = pd.MultiIndex.from_product(
    [['A', 'B'], ['one', 'two', 'three']],
    names=['letter', 'number']
)
print("\nMultiIndex din product:")
print(index_product)

# 17.4 Selecție în MultiIndex
print("\nSelecție nivel 'A':")
print(df_multi.loc['A'])

print("\nSelecție specifică:")
print(df_multi.loc[('A', 'one')])

print("\nSelecție cu slice:")
print(df_multi.loc[('A', 'one'):('B', 'two')])

# 17.5 Cross-section (xs)
print("\nCross-section nivel 'one':")
print(df_multi.xs('one', level='second'))

# 17.6 Swaplevel
print("\nSwap levels:")
df_swapped = df_multi.swaplevel('first', 'second')
print(df_swapped)

# 17.7 Stack și Unstack cu MultiIndex
df_stacking = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['X', 'Y', 'Z'])

print("\nDataFrame original:")
print(df_stacking)

stacked = df_stacking.stack()
print("\nStacked (MultiIndex):")
print(stacked)

unstacked = stacked.unstack()
print("\nUnstacked:")
print(unstacked)

# 17.8 MultiIndex în coloane
df_multi_col = pd.DataFrame(
    np.random.rand(3, 6),
    columns=pd.MultiIndex.from_product([['A', 'B'], ['one', 'two', 'three']]),
    index=['X', 'Y', 'Z']
)
print("\nMultiIndex în coloane:")
print(df_multi_col)

print("\nSelectare coloană 'A':")
print(df_multi_col['A'])

# 17.9 Set și Reset MultiIndex
df_reset = df_multi.reset_index()
print("\nReset MultiIndex:")
print(df_reset)

df_set_multi = df_reset.set_index(['first', 'second'])
print("\nSet MultiIndex:")
print(df_set_multi)

# 17.10 Sortare MultiIndex
print("\nSortare MultiIndex:")
df_sorted = df_multi.sort_index()
print(df_sorted)

print("\nSortare descrescător nivel 'second':")
df_sorted2 = df_multi.sort_index(level='second', ascending=False)
print(df_sorted2)

# 17.11 GroupBy cu MultiIndex
df_group_multi = pd.DataFrame({
    'Region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250, 120, 180]
})

grouped_multi = df_group_multi.groupby(['Region', 'Product'])['Sales'].sum()
print("\nGroupBy rezultă MultiIndex:")
print(grouped_multi)

print("\n" + "="*80 + "\n")

# ============================================================================
# 18. OPTIMIZARE ȘI PERFORMANȚĂ
# ============================================================================

print("18. OPTIMIZARE ȘI PERFORMANȚĂ\n")

# 18.1 Alegerea tipului de date corect
print("Optimizare tipuri de date:")

df_optimize = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 10000),
    'float_col': np.random.rand(10000),
    'bool_col': np.random.choice([True, False], 10000),
    'cat_col': np.random.choice(['A', 'B', 'C', 'D'], 10000)
})

print("Memorie înainte de optimizare:")
mem_before = df_optimize.memory_usage(deep=True).sum() / 1024
print(f"{mem_before:.2f} KB")

# Optimizare int
df_optimize['int_col'] = df_optimize['int_col'].astype('int8')

# Optimizare float
df_optimize['float_col'] = df_optimize['float_col'].astype('float32')

# Optimizare categorical
df_optimize['cat_col'] = df_optimize['cat_col'].astype('category')

print("\nMemorie după optimizare:")
mem_after = df_optimize.memory_usage(deep=True).sum() / 1024
print(f"{mem_after:.2f} KB")
print(f"Economie: {(1 - mem_after/mem_before) * 100:.1f}%")

# 18.2 Vectorizare vs Iterare
print("\nVectorizare vs Iterare:")
df_speed = pd.DataFrame({'A': range(10000), 'B': range(10000)})

# Iterare (LENT)
start = time.time()
result_iter = []
for i in range(len(df_speed)):
    result_iter.append(df_speed.loc[i, 'A'] + df_speed.loc[i, 'B'])
time_iter = time.time() - start

# Apply (MAI RAPID)
start = time.time()
result_apply = df_speed.apply(lambda row: row['A'] + row['B'], axis=1)
time_apply = time.time() - start

# Vectorizare (CEL MAI RAPID)
start = time.time()
result_vect = df_speed['A'] + df_speed['B']
time_vect = time.time() - start

print(f"Iterare: {time_iter:.4f}s")
print(f"Apply: {time_apply:.4f}s")
print(f"Vectorizare: {time_vect:.4f}s")
print(f"Vectorizare este {time_iter/time_vect:.0f}x mai rapid decât iterare!")

# 18.3 Chunk processing pentru fișiere mari
print("\nChunk processing pentru CSV mari:")
chunk_example = """
# Citire în chunks
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Procesare chunk
    processed = chunk[chunk['column'] > 0]
    chunks.append(processed)

df = pd.concat(chunks, ignore_index=True)
"""
print(chunk_example)

# 18.4 Utilizare query pentru filtrare
df_query = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000)
})

start = time.time()
result_bool = df_query[(df_query['A'] > 0.5) & (df_query['B'] < 0.5)]
time_bool = time.time() - start

start = time.time()
result_query = df_query.query('A > 0.5 and B < 0.5')
time_query = time.time() - start

print(f"\nFiltrare boolean: {time_bool:.4f}s")
print(f"Filtrare query: {time_query:.4f}s")

# 18.5 Eval pentru expresii
df_eval = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000),
    'C': np.random.rand(100000)
})

start = time.time()
result_normal = df_eval['A'] + df_eval['B'] * df_eval['C']
time_normal = time.time() - start

start = time.time()
result_eval = df_eval.eval('A + B * C')
time_eval = time.time() - start

print(f"\nExpresie normală: {time_normal:.4f}s")
print(f"Expresie eval: {time_eval:.4f}s")

# 18.6 Copy vs View
print("\nCopy vs View:")
df_original = pd.DataFrame({'A': [1, 2, 3]})

# View (nu copiază date)
df_view = df_original[df_original['A'] > 1]  # Poate fi view sau copy

# Copy explicit
df_copy = df_original.copy()

print("Folosește .copy() când vrei să eviți SettingWithCopyWarning")

# 18.7 Inplace operations
df_inplace = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Fără inplace (creează copie)
df_new = df_inplace.drop('B', axis=1)

# Cu inplace (modifică original)
df_inplace_copy = df_inplace.copy()
df_inplace_copy.drop('B', axis=1, inplace=True)

print("\nInplace poate economisi memorie dar este adesea mai puțin eficient")
print("Preferă reatribuirea: df = df.drop('B', axis=1)")

# 18.8 Index optimization
print("\nIndex optimization:")
df_no_index = pd.DataFrame({
    'key': np.random.choice(['A', 'B', 'C'], 100000),
    'value': np.random.rand(100000)
})

df_with_index = df_no_index.set_index('key')

start = time.time()
result_no_idx = df_no_index[df_no_index['key'] == 'A']
time_no_idx = time.time() - start

start = time.time()
result_idx = df_with_index.loc['A']
time_idx = time.time() - start

print(f"Fără index: {time_no_idx:.4f}s")
print(f"Cu index: {time_idx:.4f}s")
print("Indexarea poate accelera selectarea!")

# 18.9 Sfaturi generale
print("\n" + "-"*80)
print("SFATURI PENTRU PERFORMANȚĂ:")
print("-"*80)
print("""
1. Folosește tipuri de date corespunzătoare (int8 vs int64, category)
2. Vectorizează operațiile - evită loop-uri și iterrows()
3. Folosește query() și eval() pentru operații complexe
4. Procesează fișiere mari în chunks
5. Folosește index pentru căutări rapide
6. Evită copii inutile - folosește view-uri când e posibil
7. Folosește categorical pentru coloane cu puține valori unice
8. Filtrează date devreme pentru a reduce volumul
9. Folosește numba sau Cython pentru operații complexe
10. Consideră Dask sau Polars pentru date foarte mari (>RAM)
""")

print("\n" + "="*80 + "\n")

# ============================================================================
# 19. VIZUALIZARE CU PANDAS
# ============================================================================

print("19. VIZUALIZARE CU PANDAS\n")

print("Pandas oferă integrare cu matplotlib pentru vizualizări rapide:")

df_plot = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=100, freq='D'),
    'valoare1': np.random.randn(100).cumsum(),
    'valoare2': np.random.randn(100).cumsum(),
    'categorie': np.random.choice(['A', 'B', 'C'], 100)
})

print("\nDate pentru vizualizare:")
print(df_plot.head())

print("\nTipuri de plot-uri disponibile:")
print("""
# Line plot
df_plot.plot(x='data', y='valoare1')
df_plot.plot(x='data', y=['valoare1', 'valoare2'])

# Bar plot
df_plot['categorie'].value_counts().plot(kind='bar')
df_plot.plot.bar(x='categorie', y='valoare1')

# Histogram
df_plot['valoare1'].plot(kind='hist', bins=20)
df_plot.plot.hist(alpha=0.5)

# Box plot
df_plot[['valoare1', 'valoare2']].plot(kind='box')

# Scatter plot
df_plot.plot.scatter(x='valoare1', y='valoare2')

# Area plot
df_plot.plot.area()

# Pie chart
df_plot['categorie'].value_counts().plot(kind='pie')

# Hexbin
df_plot.plot.hexbin(x='valoare1', y='valoare2', gridsize=15)

# KDE (density)
df_plot['valoare1'].plot(kind='kde')
""")

print("\nParametri comuni:")
print("""
- figsize=(width, height): dimensiune figură
- title='Titlu': titlu grafic
- xlabel, ylabel: etichete axe
- legend=True/False: legendă
- color: culoare
- alpha: transparență
- grid=True: grilă
- style: stil linie
- rot: rotație etichete
""")

print("\nExemplu complet:")
print("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
df_plot.plot(x='data', y=['valoare1', 'valoare2'], 
             ax=axes[0, 0], title='Time Series')

# Histogram
df_plot['valoare1'].plot(kind='hist', bins=20, 
                          ax=axes[0, 1], title='Histogramă')

# Box plot
df_plot[['valoare1', 'valoare2']].plot(kind='box', 
                                        ax=axes[1, 0], title='Box Plot')

# Bar plot
df_plot['categorie'].value_counts().plot(kind='bar', 
                                          ax=axes[1, 1], title='Categorii')

plt.tight_layout()
plt.show()
""")

print("\n" + "="*80 + "\n")

# ============================================================================
# 20. BEST PRACTICES ȘI CAZURI PRACTICE
# ============================================================================

print("20. BEST PRACTICES ȘI CAZURI PRACTICE\n")

print("BEST PRACTICES:")
print("-" * 80)
print("""
1. CITIREA DATELOR:
   - Specifică dtype-uri la citire pentru performanță
   - Folosește parse_dates pentru coloane de date
   - Citește în chunks pentru fișiere mari

2. CURĂȚAREA DATELOR:
   - Verifică și tratează valori lipsă consistent
   - Convertește tipuri de date corespunzător
   - Elimină duplicate
   - Validează date (range-uri, valori posibile)

3. MANIPULAREA DATELOR:
   - Folosește operații vectorizate
   - Evită iterrows() și itertuples() când e posibil
   - Folosește method chaining pentru cod mai clar
   - Set index pentru căutări frecvente

4. GRUPARE ȘI AGREGARE:
   - Folosește groupby eficient
   - Consideră transform vs apply vs agg
   - Named aggregation pentru claritate

5. PERFORMANȚĂ:
   - Alege tipurile de date potrivite
   - Folosește categorical pentru coloane cu puține valori
   - Evită copii inutile
   - Profită de index-uri

6. COD CURAT:
   - Method chaining pentru operații multiple
   - Nume descriptive pentru coloane
   - Comentează operații complexe
   - Verifică shape și dtypes frecvent
""")

print("\n" + "-"*80)
print("ERORI COMUNE:")
print("-"*80)
print("""
1. SettingWithCopyWarning:
   Problemă: df[df['A'] > 5]['B'] = 10
   Soluție: df.loc[df['A'] > 5, 'B'] = 10

2. Modificare DataFrame în iterație:
   Problemă: for idx, row in df.iterrows(): df.loc[idx] = ...
   Soluție: Folosește vectorizare sau apply

3. Confuzie între Series și DataFrame:
   Problemă: df['col'] vs df[['col']]
   Prima returnează Series, a doua DataFrame

4. Index nesetat după sortare:
   Problemă: df.sort_values('col')
   Soluție: df.sort_values('col').reset_index(drop=True)

5. Conversii implicite de tip:
   Problemă: Citire CSV fără dtype
   Soluție: Specifică dtype-uri explicit

6. Comparații cu NaN:
   Problemă: df['col'] == np.nan
   Soluție: df['col'].isna()

7. Folosire inplace când nu e necesar:
   Problemă: df.drop('col', inplace=True)
   Soluție: df = df.drop('col')  # Mai clar și adesea mai rapid
""")

print("\n" + "-"*80)
print("CAZURI PRACTICE:")
print("-"*80)

# Caz 1: Curățare date
print("\n1. CURĂȚARE DATE:")
df_dirty = pd.DataFrame({
    'nume': ['  Ana  ', 'ION', 'maria', 'RADU'],
    'vârstă': [25, -999, 30, 150],
    'salariu': ['3000', '4500.5', 'N/A', '5200']
})

print("Date murdare:")
print(df_dirty)

# Curățare
df_clean = df_dirty.copy()
df_clean['nume'] = df_clean['nume'].str.strip().str.title()
df_clean['vârstă'] = pd.to_numeric(df_clean['vârstă'], errors='coerce')
df_clean['vârstă'] = df_clean['vârstă'].apply(lambda x: x if 0 < x < 120 else np.nan)
df_clean['salariu'] = pd.to_numeric(df_clean['salariu'], errors='coerce')

print("\nDate curate:")
print(df_clean)

# Caz 2: Feature Engineering
print("\n2. FEATURE ENGINEERING:")
df_fe = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=10, freq='D'),
    'tranzacții': [5, 8, 3, 12, 6, 9, 4, 7, 11, 5],
    'valoare': [100, 200, 50, 300, 150, 250, 80, 180, 280, 120]
})

# Extragere features temporale
df_fe['zi_săptămână'] = df_fe['data'].dt.dayofweek
df_fe['este_weekend'] = df_fe['zi_săptămână'].isin([5, 6])
df_fe['săptămână_lună'] = df_fe['data'].dt.isocalendar().week % 4 + 1

# Features derivate
df_fe['valoare_medie'] = df_fe['valoare'] / df_fe['tranzacții']
df_fe['tranzacții_cumulative'] = df_fe['tranzacții'].cumsum()

# Lag features
df_fe['tranzacții_prev'] = df_fe['tranzacții'].shift(1)
df_fe['tranzacții_diff'] = df_fe['tranzacții'].diff()

# Rolling statistics
df_fe['tranzacții_ma3'] = df_fe['tranzacții'].rolling(window=3).mean()

print(df_fe)

# Caz 3: Transformare wide to long
print("\n3. TRANSFORMARE WIDE TO LONG:")
df_wide = pd.DataFrame({
    'ID': [1, 2, 3],
    'Nume': ['Ana', 'Ion', 'Maria'],
    'Q1_2024': [100, 150, 200],
    'Q2_2024': [110, 160, 210],
    'Q3_2024': [120, 170, 220]
})

print("Wide format:")
print(df_wide)

df_long = pd.melt(
    df_wide,
    id_vars=['ID', 'Nume'],
    var_name='Trimestru',
    value_name='Vânzări'
)

print("\nLong format:")
print(df_long)

# Caz 4: Deduplicare complexă
print("\n4. DEDUPLICARE COMPLEXĂ:")
df_dup = pd.DataFrame({
    'nume': ['Ana', 'Ana', 'Ion', 'Ion', 'Maria'],
    'data': pd.date_range('2024-01-01', periods=5, freq='D'),
    'valoare': [100, 150, 200, 200, 300]
})

print("Date cu duplicate:")
print(df_dup)

# Păstrează ultimul
df_dedup = df_dup.drop_duplicates(subset=['nume'], keep='last')
print("\nPăstrează ultimul per nume:")
print(df_dedup)

# Păstrează valoarea maximă
df_max = df_dup.loc[df_dup.groupby('nume')['valoare'].idxmax()]
print("\nPăstrează valoarea maximă per nume:")
print(df_max)

# Caz 5: Agregare complexă cu condiții
print("\n5. AGREGARE COMPLEXĂ:")
df_sales = pd.DataFrame({
    'Regiune': ['Nord', 'Sud', 'Nord', 'Sud', 'Nord', 'Sud'],
    'Produs': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Vânzări': [100, 150, 200, 180, 120, 220],
    'Cantitate': [10, 15, 20, 18, 12, 22]
})

result_complex = df_sales.groupby('Regiune').agg(
    vânzări_totale=('Vânzări', 'sum'),
    vânzări_medii=('Vânzări', 'mean'),
    cantitate_totală=('Cantitate', 'sum'),
    nr_produse=('Produs', 'nunique'),
    top_vânzare=('Vânzări', 'max')
).round(2)

print(result_complex)

print("\n" + "="*80 + "\n")

# ============================================================================
# CONCLUZIE ȘI RESURSE
# ============================================================================

print("CONCLUZIE ȘI RESURSE\n")
print("="*80)

print("""
DOCUMENTAȚIE ȘI RESURSE:
------------------------
- Pandas Documentation: https://pandas.pydata.org/docs/
- User Guide: https://pandas.pydata.org/docs/user_guide/index.html
- API Reference: https://pandas.pydata.org/docs/reference/index.html
- Pandas Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

CĂRȚI RECOMANDATE:
------------------
- "Python for Data Analysis" de Wes McKinney (creatorul Pandas)
- "Pandas Cookbook" de Matt Harrison
- "Effective Pandas" de Matt Harrison

BIBLIOTECI COMPLEMENTARE:
--------------------------
- NumPy: Operații numerice de bază
- Matplotlib/Seaborn: Vizualizare
- Scikit-learn: Machine learning
- Statsmodels: Analiză statistică
- Dask: Pandas paralel pentru date mari
- Polars: Alternativă rapidă la Pandas

PRACTICĂ:
---------
1. Kaggle datasets și competiții
2. Real World Datasets (data.gov, data.world)
3. Projece personale cu date reale
4. Contribuții open-source
5. Stack Overflow - răspunde întrebări

SFATURI FINALE:
---------------
- Pandas este esențial pentru data science în Python
- Practică regulat cu dataset-uri reale
- Citește documentația - este foarte bună
- Învață să gândești în termeni de operații vectorizate
- Explorează Pandas profiling și pandas-ta pentru productivitate
- Învață SQL - conceptele se traduc bine în Pandas
- Pentru date foarte mari, consideră Dask sau Polars

================================================================================
                    SUCCES ÎN ANALIZA DATELOR CU PANDAS!
================================================================================

Acest ghid acoperă toate aspectele fundamentale și avansate ale Pandas.
Practică exemplele, experimentează cu propriile date și construiește proiecte!

Autor: Ghid Complet Pandas
Data: 2025
Versiune: Completă și actualizată cu best practices
""")

print("="*80)
print("FINAL - Ghid Complet Pandas")
print("="*80)