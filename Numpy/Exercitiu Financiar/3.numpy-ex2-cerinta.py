import numpy as np
from numpy.conftest import dtype

# 1. Încarcă Date Financiare în Array-uri NumPy
# Cerință: Încarcă fișierul 'financial_data.csv' într-un array structurat NumPy,
# incluzând coloanele: Company (string), Quarter (string), Revenue (float), Profit (float), Expenses (float).
# Inspectează primele câteva rânduri pentru a verifica încărcarea corectă.
dtype=[
    ('Company', 'U20'), #unicode de 20 char
    ('Quarter', 'U4'),  # unicode de 4 char
    ('Revenue', 'f8'),  # float pe 8 bytes
    ('Profit', 'f8'),  # float pe 8 bytes
    ('Expenses', 'f8')  # float pe 8 bytes
]
data = np.genfromtxt('2.financial_data.csv', delimiter = ',', skip_header=1, dtype=dtype)

print("#1 Încărcare date financiare în array NumPy:\n", data, "\n")
print("Primele 5 linii din fisierul cu date financiare: \n", data[:5])

# 2. Calculează Marja de Profit pentru fiecare companie și trimestru
# Cerință: Calculează Marja de Profit = Profit / Revenue pentru fiecare rând.
# Asigură-te că împarți la zero în mod corect pentru a evita erorile.

marja = np.divide( # functie care imparte primul parametru la ale doilea
    data['Profit'],
    data['Revenue'],
    out=np.zeros_like(data['Profit']), #initializeaza cu 0-uri arrayul rezultat
    where=data['Revenue']!=0 # ne asiguram ca nu facem impartiri la 0
)
print("#2 Marja de Profit per companie și trimestru:\n", marja, "\n")
for i in range(len(marja)):
    print(f" {data['Company'][i]} - {data['Quarter'][i]} - {marja[i]}")

# 3. Calculează Rata Cheltuielilor pentru fiecare companie și trimestru
# Cerință: Calculează Rata Cheltuielilor = Expenses / Revenue pentru fiecare rând.
# Aceasta arată ce fracțiune din venituri este cheltuită pe cheltuieli.
rata_ch = np.divide( # functie care imparte primul parametru la ale doilea
    data['Expenses'],
    data['Revenue'],
    out=np.zeros_like(data['Expenses']), #initializeaza cu 0-uri arrayul rezultat
    where=data['Revenue']!=0 # ne asiguram ca nu facem impartiri la 0
)

print("#3 Rata Cheltuielilor per companie și trimestru:\n", rata_ch, "\n")

# 4. Analizează Creșterea Veniturilor Trimestriale
# Cerință: Pentru fiecare companie, calculează rata de creștere a veniturilor între trimestre consecutive:
# Rata Creșterii = (Venituri_trimestru_următor - Venituri_trimestru_curent) / Venituri_trimestru_curent
rataC = []
companies = np.unique(data['Company'])
for company in companies:
    #print(company) #elementele array ului companies
    company_mask = data['Company'] == company # imi arata care sunt inregistrarile care au legatura cu iteratorul for ului -> company
    #print(company_mask)
    revenuee = data['Revenue'][company_mask] # am preluat veniturile referitoare la company
    #print(revenuee)
    growth = np.diff(revenuee)/revenuee[:-1] # [:-1] nu ia ultima iteratie in calcul (Q1 de la comp B - Q4 al comp A
    #np.diff([Q1, Q2, Q3, Q4])  # => [Q2-Q1, Q3-Q2, Q4-Q3]
    growth = np.insert(growth, 0, np.nan) # completez prima pozitie cu ceva = nan = not a number, pentru Q1 al fiecarei companii
    print("#4 Creștere Venituri Trimestriale pentru compania ", company, ":\n", growth, "\n")
    rataC.extend(growth) # adaug la finalul array ului rataC valorile din array urile growth ale fiecarei companii
rataC=np.array(rataC)
print(rataC)
for i in range(len(rataC)):
    print(f" {data['Company'][i]} - {data['Quarter'][i]} - {rataC[i]}")

# 5. Identifică Companiile cu Creștere Ridicată
# Cerință: Determină perechi companie-trimestru unde creșterea veniturilor depășește percentila 10
# a tuturor ratelor de creștere trimestrială. Aceasta identifică companii cu tendințe de creștere peste minim.

#percentila de 10% = valoarea sub care se afla 10% din toate valorile
prag = np.nanpercentile(rataC, 10)
print(prag)
masca = rataC > prag
print(masca) # cu asta pot vedea inainte care sunt inregistrarile pe care le voi afisa
high_growth_co = data['Company'][masca]
high_growth_q = data['Quarter'][masca]
high_growth_r = rataC[masca]
for c, q, r in zip(high_growth_co, high_growth_q, high_growth_r):
    print(f"Compania: {c} are in  {q} cresterea {r}")
print("#5 Companii cu Creștere Ridicată (peste percentila 10 a creșterii veniturilor):\n",  "\n")

counter_rata_crestere = {}
for company in companies:
    mask1 = ((data['Company'] == company) & masca) # care sunt companiile care se afla intre cei 90% castigatori;
                                                   # & obligatoriu in testele cu array uri
    counter_rata_crestere[company] = np.sum(mask1)
    print(mask1)
print(np.array(list(counter_rata_crestere.items())))

# 6. Numără Trimestrele cu Creștere Ridicată pentru fiecare companie
# Cerință: Pentru fiecare companie, numără trimestrele unde creșterea veniturilor depășește pragul
# percentila 10, pentru a evalua consistența creșterii.
print("#6 Număr Trimestre cu Creștere Ridicată per companie:\n", "\n")

# 7. Evaluează Oportunitățile de Investiții
# Cerință: Clasifică companiile pe baza (1) numărului de trimestre cu creștere ridicată și (2) marja medie de profit.
# Afișează companiile în ordine descrescătoare a potențialului investițional, folosind un scor combinat.
print("#7 Clasament Companii după Potențial Investițional:\n", "\n")
