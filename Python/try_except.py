
# try-except este folosita pt a trata exceptiile astfel incat programul sa nu se opreasca brusc
# sintaxa:
# try:
#   codul care poate genera o eroare
# except:
#   codul care se executa daca apare eroarea

# try -> pui codul care s-ar putea sa dea eroare
# except -> pui ce sa faca programul daca apare acea eroare

x = 10
y = 0
#print(x/y)  # ne va da eroare, 10/0 nu se poate
# o structura simpla de try/except, unde tratam eroare "ZeroDivisionError"
try:
    print(x/y)
except:
    print("Aici a aparut o exceptie")

# o structura de try/except/finally
# ce avem in blocul "finally" se executa oricum
try:
    print(x/y)
except:
    print("Aici a aparut o exceptie")
finally:
    print("Se executa oricum")

# tratarea specifica a unei exceptii
try:
    print(x/y)
except ZeroDivisionError:
    print("Nu imparti la zero!!")


# utilizare try/except pentru a rezolva o eroare de tip "TypeError"
a = 10
b = "3"
#print(a/b)   # in cazul de fata avem eroarea "TypeError"
try:
    print(a/b)
except:
    print("A aparut o exceptie aici.")

# tratarea specifica a unei exceptii
try:
    print(a/b)
except (ZeroDivisionError,TypeError):
    print("Nu poti sa imparti tipuri de date diferite.")

# putem folosi "if" in interiorul blocului "except" pt a face verificari suplimentare
try:
    print(x/y)
except ZeroDivisionError:
    if y == 0:
        print("Nu poti imparti la zero!")
    else:
        print("Alta eroare de divizare")

# IndexError -> apare cand incerci sa accesezi un index ce nu exista
# IndexError -> se aplica la: liste, tuple, string
lista = [10,20,30]

try:
    print(lista[5])   # index 5 NU exista
except IndexError as e:
    print("Indexul este in afara listei!",e)

# "except as e" ->  sintaxa gresita
# except ZeroDivisionError as e -> sintaxa corecta
# except TypeError as e -> sintaxa corecta


# KeyError -> folosimt in cazul dictionarelor

persoane = {"Ana":15,
            "Mihai":12}
try:
    print(persoane["Ion"])
except KeyError as e:
    print("Aceasta cheie nu exista in dictionar!",e)

# ValueError
try:
    numar = int("abc")
except ValueError as e:
    print("Aici a aparut o eroare de tip ValueError:",e)

# except TipEroare as e
# TipEroare = tipul erorii pe care o prinzi (ValueError, TypeError,etc)
# as e = e contine toate informatiile despre eroare aparuta




