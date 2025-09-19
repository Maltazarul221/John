# functii in Python = blocuri de cod reutilizabile care ideplinesc o sarcina specifica
# declaram o singura data o functie si ulterior o putem apela de oricate ori dorim
# functiile in Python se definesc folosind cuvantul cheie "def" + numele functiei + paranteze () + ":"
# codul pe care il contine functia este indentat cu 4 spatii
# numele functiilor trebuie sa inceapa cu litera mica

def greet():        # am creat o functie numita "greet"
    print("Ciao")   # acesta este codul functiei

greet()             # am apelat functia definita anterior "greet" (Sa nu uitam de paranteze () atunci cand apelam functia!!)

# functie care aduna 2 numere si returneaza rezultatul
# o functie poate avea "return", dar nu este obligatoriu
def suma_numerelor(a,b):
    return a + b

print(suma_numerelor(3,7))
print(suma_numerelor(6,19))
print(suma_numerelor(4,13))

variabila = suma_numerelor(42,32)

# inmultirea a 4 numere
def inmultire(a,b,c,d):             # avem 4 argumente, deci cand apelam functia trebuie sa ii dam exact 4 argumente
    return a * b * c * d

print(inmultire(3,4,8,2))

# functie care saluta o persoana
def saluta(nume):
    print(f"Salut,{nume}")    # echivalent cu: print("Salut" + str(nume))

saluta(True)
saluta("Cristi")
saluta("Bogdan")
saluta(["Ana","Ion"])
saluta(14)

# functie care verifica daca un numar este par

def este_par (numar):
    if numar % 2 == 0:
        return f"{numar} este par"
    else:
        return f"{numar} este impar"

print(este_par(37))
print(este_par(162))

# o functie cu 4 argumente care sa returneze 2 rezultate
def calculeaza(a,b,c,d):
    ridicare_la_putere = a ** b
    adunare = c + d
    return ridicare_la_putere, adunare     # daca pui mai multe valori separate prin virgula fara paranteze, Python le pune intr-un tuple

rezultat = (calculeaza(4,5,3,9))   # in variabila creata acum "rezultat" noi vom avea ceea ce ne returneaza functia
print(rezultat[0])
print(rezultat[1])

# functie care returneaza lungimea unui string
def lungime_name(nume):
    return len(nume)

print(lungime_name("Adrian"))  # ar trebui in cazul asta sa returneze 6

# functie care sa verifice daca o lista este goala
def este_lista_goala(lista):
    if len(lista) == 0:
        return True
    else:
        return False

print(este_lista_goala([]))       # returneaza "True"
print(este_lista_goala([1,2,3]))  # returneaza "False"
print(este_lista_goala(["hello",7,5,2,3,4,5,6]))

# un exemplu de list comprehension in Python:
# vrem o lista cu patratele numerelor de la 0 la 9, dar doar pt elementele pare
# varianta 1 (fara list comprehension)
patrate = []
for i in range(10):
    if i % 2 ==0:
        patrate.append(i ** 2)
print(patrate)

# varianta 2 (cu list comprehension)
# in list comprehension ordinea este: [expresie for element in secventa if conditie]
patrate = [i ** 2 for i in range(10) if i % 2 == 0]
print(patrate)







