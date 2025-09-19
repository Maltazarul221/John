
# Nested Functions (functii in functii)
# definirea unei funciti in interiorul altei functii
# utila cand vrei ca functia interna sa fie folosita doar in interiorul functiei principale (nu global in tot fisierul.py)

# calculator de TVA
# vrem o functie care calculeaza pretul final cu TVA
# in interiorul ei, avme o functie care calculeaza doar TVA-ul, iar asta nu ne intereseaza in afara functiei principale

def calculeaza_pret_final(pret, tva_procent):

    def calculeaza_tva(suma):         # NU POTI apela functia "calculeaza_tva" direct in afara functiei "calculeaza_pret_final"
        return suma * tva_procent / 100

    tva = calculeaza_tva(pret)
    print(calculeaza_tva(100))
    pret_final = pret + tva
    return pret_final

print(calculeaza_pret_final(325,21))

# daca vrem aceasta functie sa o folosim global, trebuie sa o declaram global (nu in alta functie)
def calculeaza_tva(suma,tva_procent):
    return suma * tva_procent / 100
print(calculeaza_tva(100,19))

# functii lambda
# functii lambda = este o functie anonima (fara nume)
# sintaxa = lambda argumente : expresie
# 1. nu are nevoie de return - expresia e returnata automat
# 2. poate avea oricati parametrii, dar doar singura expresie

# functie normala
def aduna(x,y):
    return x + y

# functie lambda echivalenta
aduna_lambda = lambda x,y: x + y
print(aduna(3,4))
print(aduna_lambda(3,4))

# lambda intr-un exemplu cu list + map
numere = [1,2,3,4,5,6,7,8,9,10]
patrate = list(map(lambda x : x**2,numere))
patrate_2 = list(map(lambda x : x**2,[1,2,3,4,5,6,7,8,9,10]))
print(patrate)
# map e o functie built-in si aplica functia lambda pentru fiecare element din al doilea argument

# exemplu clasic, fara lambda
def patrat(x):
    return x ** 2

rezultat = list(map(patrat,numere))
print(rezultat)

print(list(map(lambda x: x**2,[1,2,3,4])))  # Sintaxa: lambda paratametrii: expresie, argumente

# lambda + filter
# doar numere pare
# functia lambda de mai jos returneaza True doar pt numerele pare
# filter pastreaza doar elementele pt care functia returneaza True
pare = list(filter(lambda x: x % 2 == 0, numere))
print(pare)

calcul = list(filter(lambda x: x * 5 < 35 and x * 3 > 22,numere))
print(calcul)
# and,or,not = operatori logici in Python, care lucreaza cu valori booleene True/False

print((lambda x,y,z: x * y * z)(4,12,7))
print(f"Acesta este un text {(lambda x,y,z: x * y * z)(4,12,7)}")
# echivalent cu codul de mai jos:
x = (lambda x,y,z: x * y * z)(4,12,7)
print(x)

# lambda + sorted
# sorted = functie built-in care returneaza o lista sortata a elementelor
# argumentul key = lambda x:x inseamna ca fiecare el x din lista ca fi considerat tot x pt sortare (exact asa cum e el)
numere_2 = [5,2,9,1,7]
sortare_desc = sorted(numere_2,key = lambda x: x ** 2, reverse=True)
print(sortare_desc)

numere_3 = [16, 31, 4, 12, 27]
# lambda spune ca numerele pare primesc cheia 0, iar cele impare cheia 1, deci sorted le pune pe cele pare
# daca e par, lambda returneaza 0, altfel returneaza 1
# sorted sorteaza elementele astea dupa valorile returnate de lambda
# 0 < 1, deci toat enumerele cu cheia 0 (pare) vin inainte, iar cele cu cheia 1 (impare) vin dupa
sortare = sorted(numere_3,key = lambda x: 0 if x % 2 == 0 else 1)
print(sortare)

# Sintaxa funcitie "sorted": sorted(iterable, key = None, reverse = False)
studenti = [("Liviu", 8), ("Laura", 10), ("Cristi", 5)]
studenti_sortati = sorted(studenti, key=lambda student: student[1])
print(studenti_sortati)

for i in studenti_sortati:
    primul_element = i[0]
    print(primul_element)

