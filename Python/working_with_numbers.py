import math
import random
import time

# functii comune built-in legate de numere
# valoare absoluta -> functia abs() -> totul devine pozitiv
print(abs(-5.5))
print(abs(-5))
print(abs(0))

# roturngire
print(round(3.1415,2))  # rotunjire cu 2 zecimale

# minim si maxim
print(min(10,20,30))
print(max(10,20,30))

# ridicarea la putere (echivalent cu 2**3)
print(pow(2,3))
print(2 ** 3)   # echivalent cu cel de mai sus


# sum() ne face suma tuturor elementelor
print(sum([1,2,3,4,5]))
# daca se adauga elemente float -> rezultatul va fi float

# exemple cu functii din modulul "math"
print(math.floor(4.6))  # rotunjire in jos
print(math.ceil(4.2))   # rotunjire in sus
print(math.sqrt(9))     # radacina patrata (echivalent cu ce e mai jos)
print(25 ** 0.5)  # radacina patrata (radical) -> ridicam la puterea 0.5

print(math.log(1000,10))  # rezultatul =  10 la ce putere ne da 1000?
#x = numarul pentru care calculam logaritmul
#base = baza logaritmului

# nu am pus baza, Python foloseste logaritmul natural (baza e ~ 2.718281828459045...)
print(math.log(1000))  # la ce putere "e" ne va da 1000?

print(math.exp(1))  # ne calculeaza e la puterea x, adica e^x
print(math.exp(4))  # ne calculeaza e^4 = 2.718 ^ 4 = 2.718 * 2.718 * 2.718 * 2.718
print(math.exp(0))  # orice ridicam la puterea 0 ne va da 1
print(math.exp(-1)) # ceva la puterea -1, este 1/ceva

# numere complexe au 2 parti: reala + imaginara
# forma generala in matematica: z = a + bi
# forma generala in Python:     z = a + bj
# a = partea reala
# b = partea imaginara
# i (sau j in Python) = unitatea imaginara, i^2 = -1
z = 3 + 4j
print(z)
print(z.real)
print(z.imag)
print(abs(z)) # modulul numarului complex |z| = sqrt(real^2 + imag^2) = sqrt(9 + 16) = sqrt(25) = 5.0

# functii trigonometrice
print(math.pi)
print(math.sin(math.pi/2)) # sin(pi/2) = 1
print(math.cos(0))   # cos(0) = 1
# sin si cos se pot deduce pe cadranul trigonometric
# sin(30) = 1/2
# sin(45) = sqrt2/2
# sin(60) = sqrt3/2
# sin(pi/2 - x) = cos(x)
# tan = sin/cos  (tangenta)
# cot = 1/tan = cos/sin    (cotangenta)

# math.asin(x) = arcsinusul lui x, unghiul (in radiani) pentru care sin(unghiul respectiv) = x
# math.acos(x) = arccosinusul lui x, unghiul (in radiani) pentru care sin(unghiul respectiv) = x
# pi = 3.141592...
x = 1
theta = math.asin(x)   # theta = pi/2 = 3.1415 / 2 = 1.5707..
print(theta)

# math.degrees() converteste radiani -> grade
theta_deg = math.degrees(theta)
print(theta_deg)

# folosim modulul built in "random"
print(random.randint(1,100))  # imi alge un element integer intre a si b
print(random.choice([1,2,3,4,5]))  # imi alege un element random din lista
print(random.random())   # genereaza un float aleator intre 0.0 si 1.0 (exclusiv 1.0)
print(random.uniform(1,100))  # genereaza un float aleator intre a si b

# amestecarea elementelor dintr-o lista
my_list = [1,2,3,4,5]
random.shuffle(my_list)
print(my_list)

# random.sample(population,k)
# ia un subset de k elemente unice dintr-o secventa
# population poate fi lista, range, tuple

print(random.sample(range(1,100),5))    # ne ia k valori (aici 5) din population

my_set = {10,20,30,40,50}

sample = random.sample(list(my_set),3)
print(sample)

# functii din random: sample, shuffle, choise nu merg direct pe set-uri
# daca vrem sa folosim set-uri, ar trebui sa le convertim in prima faza in liste

mixed_list = [1,2,"Alice","Bob",3.14,7,"John"]
sample_mixed = random.sample(mixed_list,2)
print("Sample din lista mixta: ", sample_mixed)

print("Printez un mesaj aici.")
time.sleep(2)   # pauza de 10 secunde
print("Printez acest mesaj (dupa sleep)")

wait = random.randint(1,10)
time.sleep(wait)
print("Print dupa sleep")



