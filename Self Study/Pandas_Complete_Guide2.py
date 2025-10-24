# ============================================================================
# 21. WINDOW FUNCTIONS AVANSATE
# ============================================================================
import time

import numpy as np
import pandas as pd

print("21. WINDOW FUNCTIONS AVANSATE\n")

# Date pentru window functions
df_window = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=20, freq='D'),
    'valoare': [10, 12, 11, 14, 13, 16, 15, 18, 17, 20,
                19, 22, 21, 24, 23, 26, 25, 28, 27, 30]
})

print("Date pentru window functions:")
print(df_window.head(10))

# 21.1 Rolling (fereastră mobilă)
print("\nRolling window (3 zile):")
df_window['rolling_mean'] = df_window['valoare'].rolling(window=3).mean()
df_window['rolling_std'] = df_window['valoare'].rolling(window=3).std()
df_window['rolling_min'] = df_window['valoare'].rolling(window=3).min()
df_window['rolling_max'] = df_window['valoare'].rolling(window=3).max()
print(df_window.head(10))

# 21.2 Rolling cu multiple funcții
print("\nRolling cu agg multiple:")
rolling_multi = df_window['valoare'].rolling(window=5).agg(['mean', 'std', 'min', 'max', 'sum'])
print(rolling_multi.head(10))

# 21.3 Rolling cu center
print("\nRolling centered (fereastră centrată):")
df_window['rolling_center'] = df_window['valoare'].rolling(window=3, center=True).mean()
print(df_window[['valoare', 'rolling_mean', 'rolling_center']].head(10))

# 21.4 Expanding (fereastră cumulativă)
print("\nExpanding window (cumulative):")
df_window['expanding_mean'] = df_window['valoare'].expanding().mean()
df_window['expanding_sum'] = df_window['valoare'].expanding().sum()
df_window['expanding_max'] = df_window['valoare'].expanding().max()
print(df_window[['valoare', 'expanding_mean', 'expanding_sum']].head(10))

# 21.5 Exponentially Weighted Moving (EWM)
print("\nExponentially Weighted Moving Average:")
df_window['ewm_mean'] = df_window['valoare'].ewm(span=3).mean()
df_window['ewm_std'] = df_window['valoare'].ewm(span=3).std()
print(df_window[['valoare', 'rolling_mean', 'ewm_mean']].head(10))

# Comparație între rolling și ewm
print("\nComparație Rolling vs EWM:")
print("Rolling dă greutate egală tuturor valorilor din fereastră")
print("EWM dă mai multă greutate valorilor recente")

# 21.6 Rolling cu funcție custom
print("\nRolling cu funcție custom:")


def range_func(x):
    return x.max() - x.min()


df_window['rolling_range'] = df_window['valoare'].rolling(window=5).apply(range_func)
print(df_window[['valoare', 'rolling_range']].head(10))

# 21.7 Rolling cu min_periods
print("\nRolling cu min_periods:")
df_window['rolling_min_periods'] = df_window['valoare'].rolling(window=5, min_periods=2).mean()
print("Începe calculul când are minim 2 valori în loc de 5")
print(df_window[['valoare', 'rolling_min_periods']].head(6))

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 22. PANDAS STYLING
# ============================================================================

print("22. PANDAS STYLING - FORMATARE PROFESIONALĂ\n")

# Date pentru styling
df_style = pd.DataFrame({
    'Produs': ['Laptop', 'Mouse', 'Tastatură', 'Monitor', 'Webcam'],
    'Vânzări': [45000, 8500, 12000, 38000, 15000],
    'Profit': [12000, 2500, 3500, 10000, 4000],
    'Marjă': [0.267, 0.294, 0.292, 0.263, 0.267],
    'Creștere': [0.15, -0.05, 0.23, 0.08, -0.12]
})

print("Date pentru styling:")
print(df_style)

print("\nExemple de styling (cod pentru Jupyter/HTML):")

styling_examples = """
# 22.1 Highlighting valori
styled = df_style.style.highlight_max(subset=['Vânzări'], color='lightgreen')
styled = df_style.style.highlight_min(subset=['Profit'], color='lightcoral')

# 22.2 Gradient de culori
styled = df_style.style.background_gradient(subset=['Vânzări'], cmap='Blues')
styled = df_style.style.background_gradient(subset=['Profit'], cmap='Greens')

# 22.3 Bare de progres
styled = df_style.style.bar(subset=['Vânzări'], color='#5fba7d')

# 22.4 Formatare numere
styled = df_style.style.format({
    'Vânzări': '{:,.0f} RON',
    'Profit': '{:,.0f} RON',
    'Marjă': '{:.1%}',
    'Creștere': '{:+.1%}'
})

# 22.5 Culori condiționate
def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return f'color: {color}'

styled = df_style.style.applymap(color_negative_red, subset=['Creștere'])

# 22.6 Styling complex
styled = (df_style.style
    .background_gradient(subset=['Vânzări'], cmap='Blues')
    .bar(subset=['Profit'], color='lightgreen')
    .format({
        'Vânzări': '{:,.0f}',
        'Marjă': '{:.1%}',
        'Creștere': '{:+.1%}'
    })
    .set_caption('Raport Vânzări - Q1 2024')
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#40466e'), 
                                     ('color', 'white'),
                                     ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
)

# 22.7 Export styled to Excel
styled.to_excel('raport_styled.xlsx', engine='openpyxl')

# 22.8 Export to HTML
html = styled.to_html()
with open('raport.html', 'w') as f:
    f.write(html)

# 22.9 Highlighting condiționat avansat
def highlight_rows(row):
    if row['Creștere'] < 0:
        return ['background-color: #ffcccc'] * len(row)
    elif row['Creștere'] > 0.15:
        return ['background-color: #ccffcc'] * len(row)
    else:
        return [''] * len(row)

styled = df_style.style.apply(highlight_rows, axis=1)

# 22.10 Chain multiple styles
styled = (df_style.style
    .highlight_max(subset=['Vânzări'], color='yellow')
    .highlight_min(subset=['Vânzări'], color='orange')
    .format({'Marjă': '{:.1%}'})
    .set_properties(**{'text-align': 'center'})
)
"""

print(styling_examples)

print("\nAvantaje styling:")
print("- Rapoarte profesionale automate")
print("- Export Excel cu formatare")
print("- Vizualizare rapidă în Jupyter")
print("- Identificare rapidă pattern-uri")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 23. MEMORY OPTIMIZATION DETALIAT
# ============================================================================

print("23. MEMORY OPTIMIZATION DETALIAT\n")

# Creăm un DataFrame mare pentru teste
n_rows = 100000
df_mem = pd.DataFrame({
    'int_col': np.random.randint(0, 100, n_rows),
    'float_col': np.random.rand(n_rows),
    'category_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
    'bool_col': np.random.choice([True, False], n_rows),
    'datetime_col': pd.date_range('2020-01-01', periods=n_rows, freq='T'),
    'string_col': np.random.choice(['value1', 'value2', 'value3'], n_rows)
})

print(f"DataFrame cu {n_rows:,} rânduri")
print("\nMemorie ÎNAINTE de optimizare:")
print(df_mem.memory_usage(deep=True))
mem_before = df_mem.memory_usage(deep=True).sum() / 1024 / 1024
print(f"\nTotal: {mem_before:.2f} MB")

# 23.1 Optimizare int
print("\n23.1 Optimizare INT:")
print(f"int_col înainte: {df_mem['int_col'].dtype}")
print(f"Range: {df_mem['int_col'].min()} - {df_mem['int_col'].max()}")

# Verificăm ce tip putem folosi
if df_mem['int_col'].max() < 128 and df_mem['int_col'].min() >= -128:
    df_mem['int_col'] = df_mem['int_col'].astype('int8')
elif df_mem['int_col'].max() < 32768 and df_mem['int_col'].min() >= -32768:
    df_mem['int_col'] = df_mem['int_col'].astype('int16')

print(f"int_col după: {df_mem['int_col'].dtype}")

# 23.2 Optimizare float
print("\n23.2 Optimizare FLOAT:")
print(f"float_col înainte: {df_mem['float_col'].dtype}")
df_mem['float_col'] = df_mem['float_col'].astype('float32')
print(f"float_col după: {df_mem['float_col'].dtype}")

# 23.3 Conversie la categorical
print("\n23.3 Conversie la CATEGORICAL:")
print(f"category_col înainte: {df_mem['category_col'].dtype}")
mem_string = df_mem['category_col'].memory_usage(deep=True) / 1024
df_mem['category_col'] = df_mem['category_col'].astype('category')
mem_cat = df_mem['category_col'].memory_usage(deep=True) / 1024
print(f"category_col după: {df_mem['category_col'].dtype}")
print(f"Economie: {(1 - mem_cat / mem_string) * 100:.1f}%")

# Similar pentru string_col
df_mem['string_col'] = df_mem['string_col'].astype('category')


# 23.4 Funcție automată de optimizare
def optimize_dataframe(df):
    """Optimizează automat tipurile de date"""
    initial_mem = df.memory_usage(deep=True).sum() / 1024 / 1024

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object' and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)

        elif col_type == 'object':
            # Conversie la categorical dacă are puține valori unice
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    final_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f'Memorie înainte: {initial_mem:.2f} MB')
    print(f'Memorie după: {final_mem:.2f} MB')
    print(f'Reducere: {100 * (initial_mem - final_mem) / initial_mem:.1f}%')

    return df


print("\n23.4 Funcție automată de optimizare:")
print("Cod disponibil în exemplu - rulează optimize_dataframe(df)")

# 23.5 Verificare finală
print("\nMemorie DUPĂ optimizare:")
print(df_mem.memory_usage(deep=True))
mem_after = df_mem.memory_usage(deep=True).sum() / 1024 / 1024
print(f"\nTotal: {mem_after:.2f} MB")
print(f"Reducere: {(1 - mem_after / mem_before) * 100:.1f}%")

# 23.6 Sfaturi memorie
print("\nSFATURI MEMORY OPTIMIZATION:")
print("""
1. int64 → int8/int16/int32 (dacă range permite)
2. float64 → float32 (pierdere minimă precizie)
3. object → category (pentru coloane cu puține valori unice)
4. Șterge coloane neutilizate
5. Citește doar coloanele necesare: pd.read_csv(usecols=[...])
6. Procesează în chunks pentru fișiere mari
7. Folosește dtype la citire: pd.read_csv(dtype={...})
8. Verifică null values - drop dacă > 50%
9. Folosește sparse pentru date cu multe zeros
10. Consideră Parquet în loc de CSV (mai eficient)
""")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 24. METHOD CHAINING - COD ELEGANT
# ============================================================================

print("24. METHOD CHAINING - COD ELEGANT ȘI PROFESIONAL\n")

# Date pentru method chaining
df_chain = pd.DataFrame({
    'nume': ['  Ana Pop  ', 'ION ionescu', 'maria POPESCU', 'RADU Vasile'],
    'vârstă': [25, -999, 30, 150],
    'salariu': [3000, 4500, 2800, 5200],
    'departament': ['IT', 'HR', 'Sales', 'IT'],
    'experiență': [2, 5, 1, 8]
})

print("Date inițiale (murdare):")
print(df_chain)

# 24.1 Fără method chaining (tradițional)
print("\nFĂRĂ method chaining (5 linii separate):")
traditional_code = """
df_temp = df_chain.copy()
df_temp['nume'] = df_temp['nume'].str.strip().str.title()
df_temp = df_temp[df_temp['vârstă'] < 120]
df_temp['salariu_anual'] = df_temp['salariu'] * 12
df_result = df_temp.sort_values('salariu', ascending=False)
"""
print(traditional_code)

# 24.2 Cu method chaining (elegant, o singură expresie)
print("\nCU method chaining (elegant):")
df_cleaned = (df_chain
              .copy()
              .assign(nume=lambda x: x['nume'].str.strip().str.title())
              .query('vârstă < 120 & vârstă > 0')
              .assign(salariu_anual=lambda x: x['salariu'] * 12)
              .sort_values('salariu', ascending=False)
              .reset_index(drop=True)
              )

print(df_cleaned)

# 24.3 Method chaining complex
print("\nMethod chaining complex - transformare completă:")
chaining_example = """
df_result = (df
    # Citire și selecție
    .pipe(lambda x: x[x['valoare'] > 0])

    # Curățare
    .assign(
        nume=lambda x: x['nume'].str.strip().str.title(),
        data=lambda x: pd.to_datetime(x['data'])
    )

    # Feature engineering
    .assign(
        an=lambda x: x['data'].dt.year,
        lună=lambda x: x['data'].dt.month,
        valoare_log=lambda x: np.log1p(x['valoare'])
    )

    # Filtrare
    .query('an >= 2023')
    .dropna(subset=['valoare'])

    # Agregare
    .groupby(['an', 'lună'])
    .agg({
        'valoare': ['sum', 'mean', 'count'],
        'nume': 'nunique'
    })
    .reset_index()

    # Sortare finală
    .sort_values(['an', 'lună'])
)
"""
print(chaining_example)

# 24.4 Pipe pentru funcții custom
print("\n24.4 Pipe pentru funcții custom:")


def remove_outliers(df, column, n_std=3):
    """Elimină outliers la n deviații standard"""
    mean = df[column].mean()
    std = df[column].std()
    return df[
        (df[column] >= mean - n_std * std) &
        (df[column] <= mean + n_std * std)
        ]


def add_category(df, column, bins, labels):
    """Adaugă categorie bazată pe binning"""
    df[f'{column}_category'] = pd.cut(df[column], bins=bins, labels=labels)
    return df


# Folosire cu pipe
df_piped = (df_chain
            .copy()
            .pipe(remove_outliers, 'vârstă')
            .pipe(add_category, 'vârstă', bins=[0, 25, 35, 100], labels=['Tânăr', 'Mediu', 'Senior'])
            .assign(salariu_k=lambda x: x['salariu'] / 1000)
            )

print(df_piped)

# 24.5 Avantaje method chaining
print("\nAVANTAJE METHOD CHAINING:")
print("""
✅ Cod mai lizibil și mai ușor de urmărit
✅ Nu poluează namespace-ul cu variabile temporare
✅ Ușor de debugat (comentezi o linie)
✅ Funcțional programming style
✅ Reduce riscul de erori (nu modifici accidental variabile)
✅ Cod mai compact și elegant
✅ Pipeline-uri clare și reutilizabile

DEZAVANTAJE:
❌ Mai greu de debugat pentru începători
❌ Stack traces mai lungi
❌ Poate fi greu de citit dacă e prea complex
""")

# 24.6 Best practices pentru chaining
print("\nBEST PRACTICES:")
print("""
1. Folosește paranteze rotunde pentru formatare pe mai multe linii
2. Un pas per linie pentru lizibilitate
3. Comentează secțiuni complexe
4. Folosește .pipe() pentru logică custom
5. Folosește lambda în .assign() pentru claritate
6. Nu exagera - dacă devine neclar, folosește variabile
7. Testează fiecare pas individual
""")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 25. CUSTOM ACCESSORS
# ============================================================================

print("25. CUSTOM ACCESSORS - EXTINDE PANDAS\n")

print("Custom Accessors permit crearea propriilor metode .str, .dt, .cat")

accessor_example = """
# 25.1 Creare custom accessor
@pd.api.extensions.register_dataframe_accessor("business")
class BusinessAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def calculate_profit(self, revenue_col, cost_col):
        '''Calculează profit'''
        return self._obj[revenue_col] - self._obj[cost_col]

    def calculate_margin(self, revenue_col, cost_col):
        '''Calculează marjă profit'''
        profit = self.calculate_profit(revenue_col, cost_col)
        return (profit / self._obj[revenue_col] * 100).round(2)

    def categorize_performance(self, metric_col, thresholds):
        '''Categorizează performanță'''
        return pd.cut(
            self._obj[metric_col],
            bins=thresholds,
            labels=['Slab', 'Mediu', 'Bun', 'Excelent']
        )

# Folosire
df = pd.DataFrame({
    'venituri': [1000, 1500, 2000, 2500],
    'costuri': [700, 1000, 1300, 1600]
})

df['profit'] = df.business.calculate_profit('venituri', 'costuri')
df['marjă%'] = df.business.calculate_margin('venituri', 'costuri')

# 25.2 String accessor custom
@pd.api.extensions.register_series_accessor("text_analysis")
class TextAnalysisAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if obj.dtype != object:
            raise AttributeError("Doar pentru Series de tip string")

    def word_count(self):
        '''Numără cuvinte'''
        return self._obj.str.split().str.len()

    def char_count(self):
        '''Numără caractere'''
        return self._obj.str.len()

    def uppercase_ratio(self):
        '''Raport majuscule'''
        upper = self._obj.str.count(r'[A-Z]')
        total = self._obj.str.len()
        return (upper / total * 100).round(2)

# Folosire
s = pd.Series(['Hello World', 'PANDAS is GREAT', 'Python Programming'])
s.text_analysis.word_count()
s.text_analysis.uppercase_ratio()

# 25.3 Numeric accessor custom
@pd.api.extensions.register_series_accessor("stats")
class StatsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def outliers(self, n_std=3):
        '''Detectează outliers'''
        mean = self._obj.mean()
        std = self._obj.std()
        return self._obj[
            (self._obj < mean - n_std * std) |
            (self._obj > mean + n_std * std)
        ]

    def normalize(self):
        '''Normalizează 0-1'''
        return (self._obj - self._obj.min()) / (self._obj.max() - self._obj.min())

    def z_score(self):
        '''Calculează z-score'''
        return (self._obj - self._obj.mean()) / self._obj.std()

# Folosire
s = pd.Series([1, 2, 3, 100, 4, 5, 6])
s.stats.outliers()
s.stats.normalize()
s.stats.z_score()
"""

print(accessor_example)

print("\nAVANTAJE CUSTOM ACCESSORS:")
print("""
✅ Cod reutilizabil și modular
✅ API consistent cu Pandas (.str, .dt)
✅ Encapsulare logică de business
✅ Ușor de testat și menținut
✅ Poate fi distribuit ca library
""")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 26. SQLALCHEMY INTEGRATION
# ============================================================================

print("26. SQLALCHEMY INTEGRATION - BAZE DE DATE AVANSATE\n")

sqlalchemy_example = """
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker

# 26.1 Creare conexiune
# SQLite (fișier local)
engine = create_engine('sqlite:///mydatabase.db')

# PostgreSQL
engine = create_engine('postgresql://user:password@localhost:5432/mydb')

# MySQL
engine = create_engine('mysql+pymysql://user:password@localhost:3306/mydb')

# SQL Server
engine = create_engine('mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server')

# 26.2 Citire date
# Simplă
df = pd.read_sql('SELECT * FROM users', engine)

# Cu parametri (safe from SQL injection)
query = "SELECT * FROM users WHERE age > :min_age"
df = pd.read_sql(query, engine, params={'min_age': 25})

# Citire tabel complet
df = pd.read_sql_table('users', engine)

# Cu condiții complexe
query = '''
SELECT 
    u.name,
    u.age,
    o.order_total,
    o.order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.age > 25
ORDER BY o.order_total DESC
'''
df = pd.read_sql(query, engine)

# 26.3 Scriere date
df = pd.DataFrame({
    'name': ['Ana', 'Ion', 'Maria'],
    'age': [25, 30, 22],
    'salary': [3000, 4500, 2800]
})

# if_exists: 'fail', 'replace', 'append'
df.to_sql('employees', engine, if_exists='append', index=False)

# Cu tip date specific
from sqlalchemy.types import Integer, String, Float
df.to_sql('employees', engine, 
          dtype={'name': String(50), 'age': Integer, 'salary': Float},
          if_exists='replace', index=False)

# 26.4 Chunk processing pentru tabele mari
chunk_size = 10000
for chunk in pd.read_sql('SELECT * FROM large_table', engine, chunksize=chunk_size):
    # Procesare chunk
    processed = chunk[chunk['value'] > 0]
    processed.to_sql('processed_table', engine, if_exists='append', index=False)

# 26.5 Transactions
from sqlalchemy import text

with engine.begin() as conn:
    # Multiple operații în aceeași tranzacție
    df1.to_sql('table1', conn, if_exists='append', index=False)
    df2.to_sql('table2', conn, if_exists='append', index=False)
    # Dacă orice operație eșuează, toate sunt rolled back

# 26.6 Creare tabele
metadata = MetaData()

users_table = Table('users', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(50)),
    Column('age', Integer),
    Column('email', String(100))
)

metadata.create_all(engine)

# 26.7 Update date
from sqlalchemy import update

with engine.begin() as conn:
    stmt = update(users_table).where(users_table.c.age < 25).values(status='junior')
    conn.execute(stmt)

# 26.8 Delete date
from sqlalchemy import delete

with engine.begin() as conn:
    stmt = delete(users_table).where(users_table.c.age > 65)
    conn.execute(stmt)

# 26.9 Optimizare performanță
# Folosește index
with engine.begin() as conn:
    conn.execute(text('CREATE INDEX idx_age ON users(age)'))

# Batch insert (mai rapid)
df.to_sql('large_table', engine, 
          if_exists='append', 
          index=False, 
          method='multi',  # Batch insert
          chunksize=1000)

# 26.10 Context manager
with engine.connect() as conn:
    df = pd.read_sql('SELECT * FROM users', conn)
    # Conexiunea se închide automat

# 26.11 Pool connections
from sqlalchemy.pool import QueuePool

engine = create_engine('postgresql://user:pass@localhost/db',
                      poolclass=QueuePool,
                      pool_size=10,
                      max_overflow=20)
"""

print(sqlalchemy_example)

print("\nBEST PRACTICES SQL:")
print("""
1. Folosește parametri pentru securitate (SQL injection)
2. Creează index-uri pentru coloane frecvent căutate
3. Procesează în chunks pentru tabele mari
4. Folosește transactions pentru multiple operații
5. Închide conexiunile când nu le mai folosești
6. Folosește connection pooling pentru aplicații
7. Optimizează query-urile SQL
8. Folosește tipuri de date specifice la to_sql()
""")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 27. FEATURE ENGINEERING PENTRU MACHINE LEARNING
# ============================================================================

print("27. FEATURE ENGINEERING AVANSAT PENTRU ML\n")

# Date pentru ML
np.random.seed(42)
df_ml = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'temperatura': np.random.normal(20, 5, 1000),
    'umiditate': np.random.uniform(30, 80, 1000),
    'vânt': np.random.gamma(2, 2, 1000),
    'precipitații': np.random.exponential(0.5, 1000),
    'categorie': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'target': np.random.binomial(1, 0.3, 1000)
})

print("Date pentru feature engineering:")
print(df_ml.head())

# 27.1 Features temporale
print("\n27.1 FEATURES TEMPORALE:")
df_ml['an'] = df_ml['data'].dt.year
df_ml['lună'] = df_ml['data'].dt.month
df_ml['zi'] = df_ml['data'].dt.day
df_ml['zi_săptămână'] = df_ml['data'].dt.dayofweek
df_ml['zi_an'] = df_ml['data'].dt.dayofyear
df_ml['săptămână_an'] = df_ml['data'].dt.isocalendar().week
df_ml['oră'] = df_ml['data'].dt.hour
df_ml['este_weekend'] = df_ml['zi_săptămână'].isin([5, 6]).astype(int)
df_ml['este_lună_vară'] = df_ml['lună'].isin([6, 7, 8]).astype(int)

# Features ciclice
df_ml['oră_sin'] = np.sin(2 * np.pi * df_ml['oră'] / 24)
df_ml['oră_cos'] = np.cos(2 * np.pi * df_ml['oră'] / 24)
df_ml['lună_sin'] = np.sin(2 * np.pi * df_ml['lună'] / 12)
df_ml['lună_cos'] = np.cos(2 * np.pi * df_ml['lună'] / 12)

print(df_ml[['data', 'oră', 'oră_sin', 'oră_cos']].head())

# 27.2 Lag features
print("\n27.2 LAG FEATURES (time series):")
for lag in [1, 2, 3, 6, 12, 24]:
    df_ml[f'temperatura_lag_{lag}'] = df_ml['temperatura'].shift(lag)

df_ml[f'temperatura_diff_1'] = df_ml['temperatura'].diff(1)
df_ml[f'temperatura_pct_change'] = df_ml['temperatura'].pct_change()

print(df_ml[['temperatura', 'temperatura_lag_1', 'temperatura_diff_1']].head(5))

# 27.3 Rolling features
print("\n27.3 ROLLING FEATURES:")
for window in [6, 12, 24]:
    df_ml[f'temp_rolling_mean_{window}'] = df_ml['temperatura'].rolling(window).mean()
    df_ml[f'temp_rolling_std_{window}'] = df_ml['temperatura'].rolling(window).std()
    df_ml[f'temp_rolling_min_{window}'] = df_ml['temperatura'].rolling(window).min()
    df_ml[f'temp_rolling_max_{window}'] = df_ml['temperatura'].rolling(window).max()

# EWM features
df_ml['temp_ewm_12'] = df_ml['temperatura'].ewm(span=12).mean()

print(df_ml[['temperatura', 'temp_rolling_mean_12', 'temp_ewm_12']].head(15))

# 27.4 Interacții între features
print("\n27.4 FEATURE INTERACTIONS:")
df_ml['temp_x_umiditate'] = df_ml['temperatura'] * df_ml['umiditate']
df_ml['temp_div_umiditate'] = df_ml['temperatura'] / (df_ml['umiditate'] + 1)
df_ml['confort_index'] = df_ml['temperatura'] - 0.55 * (1 - df_ml['umiditate'] / 100) * (df_ml['temperatura'] - 14)

# Polynomial features
df_ml['temp_squared'] = df_ml['temperatura'] ** 2
df_ml['temp_cubed'] = df_ml['temperatura'] ** 3

print(df_ml[['temperatura', 'umiditate', 'temp_x_umiditate', 'confort_index']].head())

# 27.5 Binning (discretization)
print("\n27.5 BINNING:")
df_ml['temp_bins'] = pd.cut(df_ml['temperatura'],
                            bins=[-np.inf, 10, 15, 20, 25, np.inf],
                            labels=['foarte_frig', 'frig', 'moderat', 'cald', 'foarte_cald'])

df_ml['temp_qbins'] = pd.qcut(df_ml['temperatura'],
                              q=5,
                              labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

print(df_ml[['temperatura', 'temp_bins', 'temp_qbins']].head(10))

# 27.6 Agregări per grup
print("\n27.6 AGREGĂRI PER GRUP:")
df_ml['temp_mean_per_category'] = df_ml.groupby('categorie')['temperatura'].transform('mean')
df_ml['temp_std_per_category'] = df_ml.groupby('categorie')['temperatura'].transform('std')
df_ml['count_per_category'] = df_ml.groupby('categorie')['temperatura'].transform('count')

# Target encoding (atenție la overfitting!)
target_means = df_ml.groupby('categorie')['target'].mean()
df_ml['target_enc_category'] = df_ml['categorie'].map(target_means)

print(df_ml[['categorie', 'temperatura', 'temp_mean_per_category']].head())

# 27.7 Frequency encoding
print("\n27.7 FREQUENCY ENCODING:")
freq = df_ml['categorie'].value_counts(normalize=True)
df_ml['category_freq'] = df_ml['categorie'].map(freq)

print(df_ml[['categorie', 'category_freq']].head())

# 27.8 Mathematical transformations
print("\n27.8 TRANSFORMĂRI MATEMATICE:")
df_ml['temp_log'] = np.log1p(df_ml['temperatura'] + 50)  # +50 pentru valori pozitive
df_ml['temp_sqrt'] = np.sqrt(df_ml['temperatura'] + 50)
df_ml['precip_log'] = np.log1p(df_ml['precipitații'])

# Box-Cox transformation (necesită scipy)
try:
    from scipy import stats

    df_ml['temp_boxcox'], _ = stats.boxcox(df_ml['temperatura'] + 50)
except:
    print("Scipy nu este instalat pentru Box-Cox")

print(df_ml[['temperatura', 'temp_log', 'temp_sqrt']].head())

# 27.9 Normalizare/Standardizare
print("\n27.9 NORMALIZARE/STANDARDIZARE:")

# Z-score standardization
df_ml['temp_standardized'] = (df_ml['temperatura'] - df_ml['temperatura'].mean()) / df_ml['temperatura'].std()

# Min-Max normalization
df_ml['temp_normalized'] = (df_ml['temperatura'] - df_ml['temperatura'].min()) / (
            df_ml['temperatura'].max() - df_ml['temperatura'].min())

# Robust scaling (rezistent la outliers)
median = df_ml['temperatura'].median()
q75, q25 = df_ml['temperatura'].quantile([0.75, 0.25])
iqr = q75 - q25
df_ml['temp_robust'] = (df_ml['temperatura'] - median) / iqr

print(df_ml[['temperatura', 'temp_standardized', 'temp_normalized', 'temp_robust']].head())

# 27.10 Encoding categorial
print("\n27.10 ENCODING CATEGORIAL:")

# One-hot encoding
df_encoded = pd.get_dummies(df_ml, columns=['categorie'], prefix='cat')
print(f"Coloane după one-hot: {[col for col in df_encoded.columns if col.startswith('cat_')]}")

# Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_ml['category_label'] = le.fit_transform(df_ml['categorie'])

print(df_ml[['categorie', 'category_label']].head())

print("\nFEATURE ENGINEERING COMPLETE!")
print(f"Features originale: 8")
print(f"Features după engineering: {len(df_ml.columns)}")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 28. ENCODING TECHNIQUES DETALIAT
# ============================================================================

print("28. ENCODING TECHNIQUES DETALIAT\n")

df_encode = pd.DataFrame({
    'culoare': ['roșu', 'albastru', 'verde', 'roșu', 'verde', 'albastru', 'roșu'],
    'mărime': ['S', 'M', 'L', 'M', 'S', 'L', 'M'],
    'rating': ['slab', 'mediu', 'bun', 'excelent', 'mediu', 'bun', 'excelent'],
    'vânzări': [100, 200, 150, 300, 120, 250, 350]
})

print("Date pentru encoding:")
print(df_encode)

# 28.1 One-Hot Encoding
print("\n28.1 ONE-HOT ENCODING:")
df_onehot = pd.get_dummies(df_encode, columns=['culoare'], prefix='culoare')
print(df_onehot)

# Drop first pentru a evita multicoliniaritatea
df_onehot_drop = pd.get_dummies(df_encode, columns=['culoare'], prefix='culoare', drop_first=True)
print("\nCu drop_first:")
print(df_onehot_drop)

# 28.2 Label Encoding
print("\n28.2 LABEL ENCODING:")
label_map = {'slab': 0, 'mediu': 1, 'bun': 2, 'excelent': 3}
df_encode['rating_label'] = df_encode['rating'].map(label_map)
print(df_encode[['rating', 'rating_label']])

# 28.3 Ordinal Encoding (manual)
print("\n28.3 ORDINAL ENCODING:")
size_order = {'S': 1, 'M': 2, 'L': 3}
df_encode['mărime_ordinal'] = df_encode['mărime'].map(size_order)
print(df_encode[['mărime', 'mărime_ordinal']])

# 28.4 Frequency Encoding
print("\n28.4 FREQUENCY ENCODING:")
freq = df_encode['culoare'].value_counts(normalize=True)
df_encode['culoare_freq'] = df_encode['culoare'].map(freq)
print(df_encode[['culoare', 'culoare_freq']])

# 28.5 Target Encoding (Mean Encoding)
print("\n28.5 TARGET ENCODING:")
target_means = df_encode.groupby('culoare')['vânzări'].mean()
df_encode['culoare_target'] = df_encode['culoare'].map(target_means)
print(df_encode[['culoare', 'vânzări', 'culoare_target']])

print("\nATENȚIE: Target encoding poate cauza overfitting!")
print("Soluții:")
print("- Cross-validation encoding")
print("- Smoothing/regularization")
print("- Leave-one-out encoding")

# 28.6 Binary Encoding
print("\n28.6 BINARY ENCODING (concept):")
binary_example = """
Culoare → Label → Binary
roșu    → 1     → 001
albastru→ 2     → 010
verde   → 3     → 011
galben  → 4     → 100

Avantaje: mai puține coloane decât one-hot
"""
print(binary_example)

# 28.7 Hash Encoding
print("\n28.7 HASH ENCODING:")


def hash_encoding(series, n_features=8):
    return series.apply(lambda x: hash(x) % n_features)


df_encode['culoare_hash'] = hash_encoding(df_encode['culoare'], n_features=4)
print(df_encode[['culoare', 'culoare_hash']])

# 28.8 Count Encoding
print("\n28.8 COUNT ENCODING:")
counts = df_encode['culoare'].value_counts()
df_encode['culoare_count'] = df_encode['culoare'].map(counts)
print(df_encode[['culoare', 'culoare_count']])

# 28.9 Weight of Evidence (WoE) Encoding
print("\n28.9 WEIGHT OF EVIDENCE (pentru binary classification):")
woe_example = """
WoE = ln(P(event|category) / P(non-event|category))

Util pentru:
- Binary classification
- Feature selection
- Handling missing values
"""
print(woe_example)

# 28.10 Comparison
print("\nCOMPARAȚIE ENCODING METHODS:")
print("""
Method          | Pros                      | Cons
----------------|---------------------------|---------------------------
One-Hot         | Simplu, nu presupune      | Multe coloane (curse of
                | ordine                    | dimensionality)
----------------|---------------------------|---------------------------
Label           | Compact                   | Presupune ordine
----------------|---------------------------|---------------------------
Ordinal         | Păstrează ordine          | Necesită cunoaștere domeniu
----------------|---------------------------|---------------------------
Frequency       | Simplu, compact           | Pierde informație
----------------|---------------------------|---------------------------
Target          | Putere predictivă mare    | Risc overfitting
----------------|---------------------------|---------------------------
Binary          | Mai puține dimensiuni     | Pierde interpretabilitate
                | decât one-hot             |
----------------|---------------------------|---------------------------
Hash            | Fixează dimensionalitate  | Coliziuni, pierde info
""")

print("\n" + "=" * 80 + "\n")

# ============================================================================
# 29. PANDAS OPTIONS ȘI CONFIGURARE
# ============================================================================

print("29. PANDAS OPTIONS ȘI CONFIGURARE GLOBALĂ\n")

# 29.1 Display options
print("29.1 DISPLAY OPTIONS:")

display_options = """
# Număr maxim de rânduri afișate
pd.set_option('display.max_rows', 100)
pd.options.display.max_rows = 100

# Număr maxim de coloane afișate
pd.set_option('display.max_columns', 20)

# Lățime maximă pentru afișare
pd.set_option('display.width', 200)

# Precizie pentru float
pd.set_option('display.precision', 2)

# Format float
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# Afișare complete pentru coloane largi
pd.set_option('display.max_colwidth', None)

# Expansiune reprezentare
pd.set_option('display.expand_frame_repr', False)

# Afișare memorie usage
pd.set_option('display.memory_usage', True)

# Resetare toate opțiunile
pd.reset_option('all')

# Verificare opțiune curentă
pd.get_option('display.max_rows')

# Context manager pentru opțiuni temporare
with pd.option_context('display.max_rows', 10, 'display.precision', 3):
    print(df)  # Folosește opțiunile temporare
# După bloc, revin la setările anterioare
"""

print(display_options)

# 29.2 Compute options
print("\n29.2 COMPUTE OPTIONS:")

compute_options = """
# Mod compute
pd.set_option('mode.chained_assignment', 'warn')  # 'raise', 'warn', None

# Folosire numexpr pentru evaluare
pd.set_option('compute.use_numexpr', True)

# Folosire bottleneck pentru reduceri
pd.set_option('compute.use_bottleneck', True)
"""

print(compute_options)

# 29.3 IO options
print("\n29.3 I/O OPTIONS:")

io_options = """
# Excel engine
pd.set_option('io.excel.xlsx.writer', 'openpyxl')

# HDF compresi
pd.set_option('io.hdf.default_format', 'table')

# Parquet engine
pd.set_option('io.parquet.engine', 'pyarrow')
"""

print(io_options)

# 29.4 Plotting options
print("\n29.4 PLOTTING OPTIONS:")

plot_options = """
# Backend pentru plotting
pd.set_option('plotting.backend', 'matplotlib')  # sau 'plotly', 'hvplot'

# Matplotlib backend
pd.set_option('plotting.matplotlib.register_converters', True)
"""

print(plot_options)

# 29.5 Liste complete opțiuni
print("\n29.5 LISTARE TOATE OPȚIUNILE:")
print_options = """
# Toate opțiunile disponibile
pd.describe_option()

# Opțiuni specifice
pd.describe_option('display')
pd.describe_option('display.max_rows')

# Căutare opțiuni
pd.describe_option('max_rows')
"""

print(print_options)

# 29.6 Configurare recomandată
print("\n29.6 CONFIGURARE RECOMANDATĂ:")

recommended_config = """
import pandas as pd

# Display
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.expand_frame_repr', False)

# Compute
pd.set_option('mode.chained_assignment', 'warn')
pd.set_option('compute.use_numexpr', True)
pd.set_option('compute.use_bottleneck', True)

# IO
pd.set_option('io.excel.xlsx.writer', 'openpyxl')

# Salvează într-un fișier de configurare pentru reutilizare
"""

print(recommended_config)

print("\n" + "=" * 80 + "\n")

# ============================================================================
# CONCLUZIE EXTINSĂ
# ============================================================================

print("CONCLUZIE - GHID COMPLET PANDAS EXTINS\n")
print("=" * 80)

print("""
AI ACUM UN GHID COMPLET PANDAS CU:

✅ 29 CAPITOLE DETALIATE
✅ 500+ EXEMPLE DE COD
✅ Best Practices și Optimizări
✅ Integrări Avansate (SQL, ML)
✅ Tehnici Profesionale (Method Chaining, Custom Accessors)
✅ Feature Engineering Complet
✅ Memory Optimization Detaliat
✅ Encoding Techniques Comprehensive

URMĂTORII PAȘI:
1. Practică fiecare secțiune cu date reale
2. Construiește proiecte complete
3. Contribuie la proiecte open-source
4. Participă la competiții Kaggle
5. Explorează Polars și Dask pentru scale-up

BIBLIOTECI COMPLEMENTARE ESENȚIALE:
- NumPy: Fundația calculelor numerice
- Matplotlib/Seaborn: Vizualizare
- Scikit-learn: Machine Learning
- Statsmodels: Statistică avansată
- Plotly: Vizualizare interactivă
- Dask: Pandas paralel
- Polars: Alternativă rapidă
- Modin: Speed-up automată

RESURSE CONTINUE DE ÎNVĂȚARE:
📚 Pandas Documentation: pandas.pydata.org
📚 Real Python Pandas Tutorials
📚 Kaggle Learn: kaggle.com/learn/pandas
📚 DataCamp Pandas Track
📚 YouTube: Data School, Corey Schafer

COMUNITATE:
💬 Stack Overflow - pandas tag
💬 Reddit: r/datascience, r/learnpython
💬 Discord: Python Discord, Data Science servers
💬 Twitter: #pandas, #datascience

AI TOATE INSTRUMENTELE PENTRU A DEVENI EXPERT PANDAS!
Mult succes în proiectele tale de Data Science! 🚀📊🐼

================================================================================
                    FINAL - GHID COMPLET ȘI EXTINS PANDAS
================================================================================
Data: 2025
Versiune: Completă, Extinsă și Optimizată
Capitole: 29
Exemple: 500+
Niveluri: Beginner → Advanced → Expert
""")

print("=" * 80)


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

print("\n" + "=" * 80 + "\n")

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
print(f"Economie: {(1 - mem_category / mem_object) * 100:.1f}%")

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

print("\n" + "=" * 80 + "\n")

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

print("\n" + "=" * 80 + "\n")

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
print(f"Economie: {(1 - mem_after / mem_before) * 100:.1f}%")

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
print(f"Vectorizare este {time_iter / time_vect:.0f}x mai rapid decât iterare!")

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
print("\n" + "-" * 80)
print("SFATURI PENTRU PERFORMANȚĂ:")
print("-" * 80)
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

print("\n" + "=" * 80 + "\n")

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

print("\n" + "=" * 80 + "\n")

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

print("\n" + "-" * 80)
print("ERORI COMUNE:")
print("-" * 80)
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

print("\n" + "-" * 80)
print("CAZURI PRACTICE:")
print("-" * 80)

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

print("\n" + "=" * 80 + "\n")

# ============================================================================
# CONCLUZIE ȘI RESURSE
# ============================================================================

print("CONCLUZIE ȘI RESURSE\n")
print("=" * 80)

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

print("=" * 80)
print("FINAL - Ghid Complet Pandas")
print("=" * 80)