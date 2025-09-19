# veriabila globala
# variabila globala = o variabila definita in afara oricarei functii/clase
# poate fi citita din orice functie

mesaj = "Hello!"  # variabila globala

def afisare():
    print(mesaj)   # putem citi variabila globala fara probleme

afisare()

# variabila locala
# variabila locala = o var creata in interiorul unei functii si exista cat timp se executa functia
#                  = nu poate fi accesata din afara functiei
# variabilele locale sunt stocate in memoria functiei

def salut():
    text = "Welcome!"   # variabila locala
    print(text)

salut()
#print(text)   acest cod va primi eroarea "NameError" pt ca "text" nu exista in afara functiei


# modificarea unei variabile globale intr-o functiei
# daca doar citesti o variabila globala in functie -> nu ai nevoie de "global"
# daca vrei sa o modifici -> trebuie sa spui explicit "global" + nume_variabila

# in situatia de mai jos, Python vede ca facem atribuire +=, deci considera ca aia este o var locala, dar neinitializata -> EROARE
# counter = 0
#
# def increment():
#     #counter += 1
# increment()


# folosirea cuvantului cheie "global"
# daca spui "global counter" -> practic Python stie ca te referi la variabila globala si o poate modifica

counter = 0

def increment():
    global counter # spune ca vrem sa folosim variabila globala "counter"
    counter += 1    # modifica direct variabila globala (ea era 0 initial si acum a devenit 1)

increment()
print(counter)

# variabile global mutabile si imutabile
# mutabile (list,dict) -> le poti modifica fara "global"

lista = [1,2,3]

def adauga_element():
    lista.append(4)  # merge fara global

adauga_element()
print(lista)

# evita modificarea directa a variabilelor globale din functii
# e mai bine sa returnezi o valoare si sa o reasignezi in global:

counter = 0

def increment_2(val):
    return val + 1

x = increment_2(counter)





























