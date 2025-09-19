
basket_emoticoane = ["🍎", "🍌", "🍇", "🍉", "🍎", "🍌", "🍌"]

basket = ["apple","banana","watermelon","cherry"]
print(type(basket))
print(type(basket[0]))

preturi = [12.5,7.0,4.25,10.0]

basket_preturi = {}                  # initializam un dictionar gol in care vom pune fructele + preturile
for i in range(len(preturi)):        # parcurgem de la 0 la lungimea listeei de preturi - 1
    basket_preturi[basket[i]] = preturi[i]    # pentru fiecare index i, iei fructul de la pozitia i din lista "basket"
                                              # si ii asociem pretrul de la pozitia i din lista preturi
print(basket_preturi)

for x in range(8):  # range(8) ne afiseaza de fapt de la 0 la 7 (calculeaza de la 0 la n-1)
    print(x)

# zip leaga el.0 din basket cu el.0 din preturi, apoi el.1 din basket cu el.1 din preturi...
# dict() transforma perechile acelea intr-un dicitonar
basket_preturi2 = dict(zip(basket,preturi))
print(basket_preturi2)

print(preturi)  # este ca tip de date o lista (definita mai sus)
for i in preturi:   # iteram prin toate elementele listei si le afiseaza pe rand
    print(i)

for fruct in basket:     # parcurgem toate elementele lista "basket"
    if "a" in fruct:     # verificam daca o conditie este respectata
        print(fruct)

# bucla "for" e folosita pt a itera printr-o secventa (lista, tuple, string, dict, range, set) si executa un bloc de cod pt fiecare element

# un set pe care l-am parcurs cu un "for"
fructe = {"mar","banana","portocala"}
for i in fructe:
    print(i)

# un string pe care l-am parcurs cu un "for"
text = "Python"
for litera in text:
    print(litera)

# o lista de tuples pe care o parcurgem cu un "for"
perechi = [(1,"a"),(2,"b"),(3,"c")]   # o lista cu 3 elemente (index 0, index 1, index 2)
for i,j in perechi:
    print(i,j)   # ne va afisa prima data "1 a", apoi "2 b", apoi "3 c"


# un dictionar (are chei si valori) pe care il parcurgem cu un for
individ = {
    "nume":"Mircea",
    "varsta":31
}

print(individ)
print(individ.items()) # metoda ".items() returneaza un obiect iterabil ce contine perechi(cheie,valoare)

for cheie,valoare in individ.items():
    print(cheie,valoare)

# o lista simpla, pe care o parcurgem cu un "for"
for i in [0, 1, 2.56, "Python", 4]:
    print(i)


# Bucla while
# bucla while executa codul cat timp o conditie este adevarata
# while conditie: continutul buclei
# conditia = expresie booleana care controleaza bucla (Daca este True, codul continua)

x = 0          # x este o variabial de tip "int"
while x > 5:   # incepem o bucla "while" si conditia buclei este "x<5"
    print(x)
    x += 1     # este echivalent cu "x = x + 1"

# x = 0 -> 0<5 -> afiseaza 0 -> x devine 1
# x = 1 -> 1<5 -> afiseaza 1 -> x devine 2
# x = 2 -> 2<5 -> afiseaza 2 -> x devine 3
# x = 3 -> 3<5 -> afiseaza 3 -> x devine 4
# x = 4 -> 4<5 -> afiseaza 4 -> x devine 5
# x = 5 -> 5<5 -> Bucla "while" se opreste

x = 0          # x este o variabial de tip "int"
while x < 5:   # incepem o bucla "while" si conditia buclei este "x<5"
    x += 1     # este echivalent cu "x = x + 1"
    print(x)

# x = 0 -> 0<5 -> x devine 1 -> afiseaza 1
# x = 1 -> 1<5 -> x devine 2 -> afiseaza 2
# x = 2 -> 2<5 -> x devine 3 -> afiseaza 3
# x = 3 -> 3<5 -> x devine 4 -> afiseaza 4
# x = 4 -> 4<5 -> x devine 5 -> afiseaza 5
# x = 5 -> 5<5 -> Bucla "while" se opreste

animals = [
    {"name": "Mango", "type": "dog", "age": 7},    # animals[0]
    {"name": "Berry", "type": "hamster", "age":12},    # animals[1]
    {"name": "Tom", "type": "cat", "age":7}        # animals[2]
]

for animal in animals:
    if animal["type"] == "dog":
        print("Este un caine.")
    elif animal["type"] == "cat":
        print("Este o pisica.")
    else:
        print("Nu e nici caine, nici pisica.")

# in exemplul de mai jos avem un for care contine un "if" iar acesta la randul sau contine un alt "if"
for animal in animals:     # in cazul nostru (lista are 3 elemente) acest "for" se va parcurge de 3 ori
    if animal["type"] == "dog":
        print(f"{animal['name']} este un caine.")
        if animal["age"] > 10:
            print(f"{animal['name']} este un caine batran.")
        else:
            print(f"{animal['name']} este un caine tanar.")

    elif animal["type"] == "cat":
        print(f"{animal['name']} este o pisica.")
    else:
        print(f"{animal['name']} nu e nici caine, nici pisica, ci este {animal['type']} si are varsta de {animal['age']} ani.")


# range() creeaza o secventa de numere (ex.0,1,2,3,4,5..) iar "for" stie sa ia fiecare element din acea secventa automat
# while + range nu se foloseste direct, dar se poate folosi cum este in exemplul de mai jos
# while range(5) -> Intrii in bucla infinita, nu e corect!!
# este recomandat la "range" sa se foloseasca impreuna cu "for"
i = 0
while i in range(5):
    print(i)
    i+=1

x = 0
print(x)
animals_2 = ["dog", "cat", "bird", "dog", "cat"]  # lista cu 5 elemente ( index 0, index 1, index 2, index 3, index 4)

while x < (len(animals_2)):      # while x < 5 (in cazul nostru)
    animal = animals_2[x]        # am creat o varibila noua (animal), iar el. din lista aflat la pozitia x il punem in variabila
    if animal == "dog":
        print(f"Animalul {x} este un caine.")
    elif animal == "cat":
        print(f"Animalul {x} este o pisica.")
    else:
        print(f"Animalul {x} nu este nici caine, nici pisica")
    x += 1                              # incrementarea este necesara (daca nu faceam asta, x ramanea 0 si intram in bucla infinita)

print(x)

# o bucla "while" ce contine un "if" care contine la randul sau inca un "if"
x = 0
while x <= 100:    # ruleaza cat timp x este mai mic decat 100 (bucla ruleaza de la 0 pana la 100), deci de 101 ori
    if x % 2 == 0:   # daca impartit la 2 are restul 0, este par
        print(f"{x} este par.")
        if x > 50:
            print(f"{x} este par si mai mare decat 50")
    else:
        if x > 85:
            print("Facem o oprire cu break.")
            break
        print(f"{x} este impar")
    x += 1


# break intrerupe complet executia unei bucle (for sau while) imediat ce este intalnita
# apoi se va merge direct in afara buclei, ignorand iteratiile ramase
# functia "input" ne da posibilitatea sa introducem de la tastatura content

fruits = ["apple","banana","cherry","orange"]
fruits_to_find = input("Introdu un fruct pe care vrei sa-l cauti:")
fruits_found = []     # este o lista empty

for fruit in fruits:                          # se parcurge lista "fruits" cu o bucla "for"
    fruits_found.append(fruit)
    print(f"Verificam fructele {fruit}")      # la fiecare iteratie, se afiseaza elementul respectiv din lista
    if fruit == fruits_to_find:                     # daca elementul este egal cu string-ul "cherry", se afiseaz mesajul de mai jos
        print(f"Am gasit {fruit} in lista! Ma opresc aici.")
        print("Afisam toate fructele pana acum:",",".join(fruits_found))
        break                                 # daca acea conditie din "if" este respectata, se opreste automat bucla "for"

# ",".join(fruits_found) inseamna: ta toate elementele din fruits_found(care e lista) si le uneste intr-u sir de caractere,
# acele caractere sunt separate de "," in cazul asta (punem pune orice alt delimitator)


# exemplu cu while + break

number = 1

while number <=10:
    print(f"Numarul curent: {number}")
    number += 1
    if number == 5:
        print("Am ajuns la 5, opresc bucla")
        break






