
# decorator = e o metoda de a "impacheta" (wrapped) o functie pentru a-i adauga functionalitate
#             fara sa modificam codul acelei functii direct

# sintaxa:
# def "nume_decorator":
# ...
# @nume_decorator
# def nume_functie:
# ...
# nume_functie()

# aici am creat decoratorul:
def decoratorul_meu(functie_originala):      # decoratorul primeste o functie ca argument
    def functie_noua():                      # definim o fct interna care face "wrapperd" pe fct originala
        print("Inainte de functie.")         # cod care ruleaza inainte de functia originala
        functie_originala()                  # apelam functia originala
        print("Dupa functie.")               # codul care ruleaza dupa functia originala
    return functie_noua                      # returnam functia interna, fara paranteze, pentru a fi apelata mai tarziu

# aici utilizez decoratorul creat mai sus:
@decoratorul_meu
def salut():
    print("Salutare tuturor!")

# am apelat functia salut
# s-a executat tot ce are functia salut in interiorul ei + functionalitatea data de decorator
salut()


def decorator(filling_function):
    def taco():
        print("🌮 Pun sos înainte...")
        filling_function()
        print("🌮 Mai pun și salată după.")
    return taco

@decorator
def adauga_carne():
    print("🥩 Carne gătită.")

adauga_carne()

# un exemplu de decorator care dubleaza rezultatul unei functii care returneaza un numar
def dubleaza_rezultatul(functie):
    def functie_noua():
        rezultat = functie()
        return rezultat * 2
    return functie_noua

@dubleaza_rezultatul
def numar():
    return 5

print(numar())

@dubleaza_rezultatul
def functie_string():
    return "Hello"

print(functie_string())

# cand Python intalneste @decorator, decoratorul este apelat imediat, dar inca nu ruleaza inca fct originala
# decoratorul primeste fct ca argument si returneaza o functie noua (wrapped)
# cand apelezi fct decorata, functia nou creata de decorator se executa

# De ce e util decoratorul?
# separa responsabilitatile - decoratorul se ocupa de lucruri aditionale
# reutilizare - 1 singur decorator poate fi folosit pt mai multe functii
# curatenie in cod - nu bagi peste tot aceleasi bucati repetitive
# poti activa/dezactiva anumite comportamente doar punand sau scotand @nume_decorator


# *args - permite functiei sa primeasca orice nr de argumente pozitionale
# se scrie cu "*" inainte
# in interiorul functiei, args devine un tuple cu toate argumentele primite

# suma pentru 2 numere
def sum2(a,b):
    return a + b

print(sum2(10,20))

# suma pentru 3 numere
def sum3(a,b,c):
    return a + b + c

print(sum3(10,20,30))

# suma pentru 4 numere
def sum4(a,b,c,d):
    return a + b + c + d

print(sum4(10,20,30,40))

# args este argument pe care il primeste functia
# *args este un tuple ce contine toate argumentele pozitionale primite de functie

def suma(*args):
    return sum(args)

print(suma(10,20))
print(suma(10,20,30,40,50,60,70,400))

# **kwargs permite functiei sa primeasca orice nr de argumente denumite
# se scrie cu "**" inainte de nume
# in interiorul funcitiei, kwargs devine un dictionar cu perechi: cheie:valoare

def prezentare(**kwargs):
    for cheie,valoare in kwargs.items():
        print(f"{cheie}:{valoare}")

# dict.items() returneaza toate perechile cheie:valoare ale unui dictionar
# e folosita pt a itera printr-un dictionar

# apelam functia cu diferite argumente numite
prezentare(nume = "Mircea",varsta = 31)
prezentare(marca = "Tesla", model = "Model 3", anul = 2023)

# **kwargs -> permite functiei sa primeasca un numar variabil de argumente numite(cheie:valoare)

# Decorator ce foloseste *args pt a putea prelua orice nr de arguemnte de la functia dorita:

def decorator_3(functia_originala):
    def functie_noua(*args):
        print("Primul mesaj.")
        rezultat = functia_originala(*args) # aici se calculeaza noua variabila "rezultat" si ea e 8 (nu se afiseaza aici)
        print(rezultat)      # daca pun acest print -> se va afisa aici valoarea variabile "rezultat"
        print("Al doilea mesaj")
        return rezultat
    return functie_noua

@decorator_3
def aduna(a,b):
    return a + b

print(aduna(5,3))


# generator
# generator = o functie speciala care produce valori pe rand, pe masura ce ele sunt cerute,
#             in loc sa genereze toate valorile deodata si sa le stocheze in memorie
# o functie normala in python -> folosim "return"
# un generator -> folosim "yield" (in loc de "return")

# un generator foloseste memoria mai eficient decat o functie normala

# definirea unui generator (folosim "yield" in loc de "return")
def numere_pare(n):
    for i in range(n):
        if i % 2 == 0:
            yield i

for numar in numere_pare(10000):
    print(numar)

# yield "tine minte" unde a ramas functia, astfel incat la urmatoarea iteratie sa continue de unde a ramas
# nu ocupa multa memorie, pt ca nu genereaza toate valorile deodata

# cand sa folosesti generatori?
# datele sunt foarte mari: fisiere mari, milioane de valori
# economisesti memorie

# acelasi exemplu doar ca e cu "return":
def numere_pare2(n):
    lista = []
    for i in range(n):
        if i % 2 == 0:
            lista.append(i)
    return lista

rezultat_nou = numere_pare2(10000)
print(rezultat_nou)
# toate valorile sunt stocate in memorie in lista


# yield emite cate un element pe rand, fara sa stocheze 100000 in memorie
# filtrarea if < 5 face ca doar 5 elemente sa fie produse
def elemente_mici(n):
    for i in range(n):
        if i < 5:
            yield i

for e in elemente_mici(100000):
    print(e)

gen = elemente_mici(100)
# next() il folosim pt a lua elementele unul cate unul
print(next(gen))
print(next(gen))
print(next(gen))

# daca apelam next() dupa ce generatorul s-a terminat, apare StopIteration


















