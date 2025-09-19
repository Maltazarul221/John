import math  # math este un modul built in in Python (doar trebuie importat in fisierele .py unde il folosim)
import random  # random este un modul built in in Pyhton

# Python este limbaj compilat si interpretat
# Pas 1 : compilarea - fisierul .py este transformat intr-un fisier .pyc
# Pas 2 : interpretarea - codul este rulat de Python VM (Python Virtual Machine)

# Python este un "high level language" - sintaxa simpla, apropiata de limba engleza
# Python este "dynamic typed" - nu trebuie sa declari tipul variabilei
# x = 5 (Python stie automat ca este int),  y = "Salut" (Python stie automat ca este string)

# Nu se foloseste ";" ca sa inchidem o expresie (ca in C)
# Comentarii in python se pun folosind "#"

print ("Azi avem curs de Python")

x = 1 + 2 + 3 + \
    4 + 5
print (x)
print(type(x))

# tipul de date None Type
y = None
print(type(y))

# tipul de date "int"
x = 1000
print(x)

# operatii aritmetice
# + adunare
# - scadere
# * inmultire
# ** ridicarea la putere
# / impartire reala (float) - se fac eimpartirea si daca ne da 4.38, se ia 4.38
# // impartire intreaga (doar partea intreaga) - se face impartirea si daca ne da 4.38, se ia doar 4
# % restul impartirii (modulo) - se face impartirea, nu conteaza cat da, se ia doar restul
# () paranteze rotunde - singurele acceptate in expresii matematice in Python!

# = este operator de asignare (nu este operator aritmetic sau de comparatie)
# == >= <= > < sunt operatori de comparatie (nu sunt operatori aritmentici sau de asignare)

rezultat_1 = 4 * 5 + 3  # 20 + 3 = 23 (in python se respecta ordinea operatilor)
print(rezultat_1)

rezultat_2 = (9-12) + 2 * (3+6)
print(rezultat_2)

# toate cele 3 de mai jos sunt variabile de tip "int"
a = 5
b = -8
c = 0

rezultat_3 = 15 / 4   # rezultatul e "float"
print("Aici avem rezultatul impartirii (float): ")
print(rezultat_3)

rezultat_4 = 15 // 4  # rezultatul e "int"
print(rezultat_4)

rezultat_5 = 15 % 4   # aici ne afiseaza restul impartirii
print(rezultat_5)

rezultat_6 = 2 ** 5   # 2**5 = 2 * 2 * 2 * 2 * 2
print(rezultat_6)

rezultat_7 = 9 ** 0.5   # cand ridicam la puterea 0.5 este de fapt radacina patrata a acelui nr (radical)
print(rezultat_7)

print(math.sqrt(9))
print(math.sqrt(25))
print(math.pi)
print(math.cos(0))

numar = random.randint(1,100)
print(numar)

# String
# putem folosi ' ' sau " " pentru a defini un string

text1 = 'This is a single-quoted string'
print(text1)
print(type(text1))    # type e o functie built in, ne ajuta sa vedem tipul de date

text2 = "This is a double-quoted string"
print(text2)
print(type(text2))

# text3 = "Do not mix it!'  nu este o sintaxa corecta

text4 = "this " + "and that"    # concatenare de string-uri
print(text4)
text5 = text1 + " " + text2
print(text5)

# Boolean
cleaned_my_room = True
cleaned_entire_house = False

print(cleaned_my_room)
print(type(cleaned_my_room))

print(cleaned_entire_house)
print(type(cleaned_entire_house))

print(abs(-5))   # returneaza valoarea absoluta (fara semn)
print(bin(73213232132))   # reprezentarea binara
print(hex(73213232132))   # reprezentarea hexazecimal

x = "hello"   # reprezentarea binara a unui string
for char in x:
    print(f"{char} -> {ord(char)} -> {bin(ord(char))}")

print(len(x))   # len = functie built in cu ajutorul careia vedem lungimea
