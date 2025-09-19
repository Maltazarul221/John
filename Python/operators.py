
# operatori aritmetici de baza, deja cunoscuti:
# + Adunare
# - Scadere
# * Inmultire
# / Impartire
# % Modulo (restul impartirii)

# exemple cu modulo
print(15 % 3)
x = 15
y = 3
print(x % y)
print(15 % 2)  # 15/2 = 7 (restul care ne rame este 1)

# exemple cu ridicarea la putere
print(2 ** 3)  # 2 la puterea a3a
print(3 ** 5)  # 3 la puterea a5a

# impartirea intreaga (floor division)
print(7 // 2)  # 7/2 = 3.5 -> rezultatul impartirii intregi = 3

# operatori de atribuire (assignment operators)
# sunt folositi pt a atribui valori variabilelor
# cel mai simplu operator de atribuire este "="
# alti operatori de atribuire compusi ar fi "+=" "-=" "*=" "/="
my_age = 30
print(my_age)

my_age +=1  # este echivalent cu "my age = my age + 1"
print(my_age)

my_age -=2  # este echivalent cu "my age = my age - 2"
print(my_age)

my_age *=3  # este echivalent cu "my age = my age * 3"
print(my_age)

my_age /=2  # este echivalent cu "my age = my age / 2"
print(my_age)

# Operator XOR (se foloseste ^)
# "^" nu se foloseste pentru ridicarea la putere in Python!!
a = 5 # 0101
b = 3 # 0011
print(a ^ b)

# operatorii de comparatie
# sunt folositi pentru a compara valori
print(4 == 4) # True
print(4 != 4) # False ("!=" = diferit de..)
print(5 > 3)  # True
print(2 < 1)  # False
print(5 >= 5) # True
print(3 <= 2) # False

# operatorii logici (and, or, not)
# combina conditii booleene

# in cazul "and" toate conditiile trebuie sa fie "True" pt rezultat "True"
# in cazul "and" daca una din conditii este "False", rezultatul va fi "False"
print(4 == 4 and 3 == 3 and 6 == 6) # True

# in cazul "or", daca una dintre conditii este "True" -> rezultatul este "True"
# in cazul "or" daca nicio macar o conditie nu este "True" -> rezultatul este "False"
print(4 == 4 or 3 == 4)

print(not (4 == 4))  # not(True) = False
print(not (4 == 3))  # not(False) = True

# compararea structurilor de date
# putem compara structuri gen: liste, tuple sau dict

# declaram 2 variabile de tip "dict"
my_data = {"age":18}
your_data = {"age":28}

print(my_data == your_data)  # False
print(my_data["age"] == your_data["age"])  # False

my_list = [12,14,14,25]
your_list = [12,14,25]

print(my_list == your_list)   # False (listele sunt diferite)
print(my_list[2] == your_list[1])   # True (elemenetul de la index 2 din "my_list" = el de la index 1 din "your_list"





