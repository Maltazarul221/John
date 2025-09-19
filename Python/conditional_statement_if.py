from access_update_data_types import my_shopping_list

# o instructiune simpla "if" (care nu contine "else")
# sintaxa este "if conditie: ce se va executa daca acea conditie este True"
if 1 == 1:
    print("Conditia este adevarata")

# acest cod nu se va executa (conditia este Falsa si nu avem o structura if/else)
# if 1 == 2:
#     print("Conditia este adevarata")

# instructiunea "else" ofera un bloc alternativ de cod, executat daca conditia din "if" nu este indeplinita

if "hello" == "ciao":         # conditia de aici este "False", deci se va merge pe "else"
    print("Conditia este adevarata!")
    print("In if pot avea mai multe print-uri")
else:
    print("Conditia este falsa!")

print("Hello")

x = 15
y = 10

if x > 10:          # conditia este True, deci ne va afisa ce avem in instructiunea "if"
    print("x este mai mare decat 10")

if x > y:           #  conditia este True, deci ne va afisa ce avem in instructiunea "if"
    print("x este mai mare decat y")

# compararea valorilor din dictionare
person1 = {
    "name": "Anna",
    "age": 19
}
person2 = {
    "name": "Nora",
    "age": 19
}

if person1["age"] == person2["age"]:
    print("Anna si nora au aceeasi varsta.")
else:
    print("Anna si Nora nu au aceeasi varsta.")

# operatori logici (and, or, not)

if person1["age"] == person2["age"] and person1["name"] != person2["name"]:
    print("Au aceeasi varsta, dar nume diferite.")

# a2a varianta de rezolvare: conditii extrase in variabile, pentru claritate
# same_age si different_name sunt de tip "boolean"
same_age = person1["age"] == person2["age"]
different_name = person1["name"] != person2["name"]

if same_age and different_name:
    print("Au aceeasi varsta, dar nume diferite")


# compararea elementelor dintr-o lista
my_shopping_list = ["apple","apple"]

if my_shopping_list[0] == my_shopping_list[1]:
    print("Primele 2 produse din lista sunt identice")


# conditii multiple
x = 7

if x % 3 == 0 and x % 5 == 0:
    print("x este multiplu de 3 si 5")
elif x % 3 == 0:
    print("x este multiplu de 3")
elif x % 5 == 0:
    print("x este multiplu de 5")
else:
    print("x NU este multiplu de 3 sau 5")


# verificari conditionale in liste si dict
# animals = o lista care are 2 dict (fiecare dict are 3 perechi cheie/valoare)
animals = [
    {
        "name": "Mango",
        "type": "dog",
        "age": 7
    },
    {
        "name": "Berry",
        "type": "dog",
        "age" : 12
    },
    {
        "name": "Mango",
        "type": "cat",
        "age": 7
    }
]
print(animals)

if animals[0]["type"] == "dog":
    print("Primul animal este un caine.")
else:
    print("Primul animal NU este un caine.")

variabila = 250

# exemplu mai complex cu lista de dictionare
# in cazul de mai jos (unde avem "and") trebuie ca toate conditiile sa fie "True", daca nu merge pe instructiunea "else"
if animals[0]["type"] == "dog" and animals[1]["type"] == "dog" and animals[2]["type"] == "dog":
    print("Toate animalele din lista sunt caini(dog).")
else:
    print("Nu toate animalele din lista sunt caini.")




