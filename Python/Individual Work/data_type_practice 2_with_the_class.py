# Structura Date
#Cum accesam  Valori in structura de date IN DICTIONAR

#1.in python putem accesa valori folosind metoda "get"

my_personal_data = {
    "name": "John Doe",
    "age": 120
}
my_personal_data.get("name")
my_personal_data.get("age")

print(my_personal_data.get("name"))
print(my_personal_data.get("age"))

#2. Folosind parantezele patrate (liste)

my_personal_data ["name"]
my_personal_data ["age"]

print(my_personal_data["name"])
print(my_personal_data["age"])

#diferenta dintre cele 2 metode este cum manevreaza ele lipsa de chei ( date ) vezi exemplu mai jos

#print(my_personal_data.get("sex"))
#print(my_personal_data["sex"])

# metoda "get" este folositoare in cautarea de date non-critice unde datele lipsa pot fi manevrate cu usurinta
# metoda parantezelor patrate este folositoare atunci cand trebuie sa ne asiguram de prezenta datelor critice in special
#validarea lor.

#Cum facem update in structura de date IN DICTIONAR
# Putem modifica valorile in dictionar folosind metoda operator "="

my_personal_data = {"name": "John Doe", "age": 121}

my_new_age=my_personal_data["age"]

print(my_new_age)

# Cum accesam elemete din LISTE
# Putem accesa elemente din liste folosind metoda de index ( in Python numaratoarea incepe de la 0 ) sau metoda parantezelor patrate

my_shopping_list =["banana", "beer", "pills"]

my_first_item = my_shopping_list[0]
my_second_item= my_shopping_list[1]

print(my_first_item)
print(my_second_item)

#Pentru a accesa ultimul element din lista folosim:

last_item= my_shopping_list[-1]
print(last_item)

#Pentru a schimba un element dintr-o lista :

my_shopping_list[0]= "am uitat ce a zis sotia"
my_first_item= my_shopping_list [0]

print(my_first_item)

# Accesarea de elemente dintr-un dictionar integrat intr-o lista
# Daca avem dictionare integrate intr-o lista, putem accesa elementele acestora combinand indexuri-le

my_people_list= [
    {
        "name": "John",
        "age": 22

    },
    {
        "name":"Doe",
        "age":22.232
    }
]

my_first_person_name= my_people_list[0]["name"]
my_first_person_age=my_people_list[0]["age"]
my_second_person_name= my_people_list[1]["name"]
my_second_person_age= my_people_list[1]["age"]

print(my_first_person_name)
print(my_first_person_age)
print(my_second_person_name)
print(my_second_person_age)

my_people_list[0]["name"]= "This has nothing to do with the above codes :D" # asta o sa faca update la nume
my_first_person_name= my_people_list[0]["name"]

print(my_first_person_name)

# Cum accesam datele din seturi

# In python seturile sunt o colectie de date aleatori si unice, prin urmare nu pot fi accesate ca si listele sau tuple prin index.
# Utilizarea principala a seturile este pentru a testa apartenenta lor

my_set= {"apple","cow", "sheep"}
exists= "apple" in my_set  # aici verificam daca apple exista in set

print(exists)

# sa inceram ceva ce nu este in lista
exists= "alien" in my_set
print(exists)

# Cum adagam date in seturi
# Pentru a adauga date in set folosim comanda ".add" ( se adauga elementul specificat daca el nu este deja adaugat in set )

my_set= {"apa","paie", "bataie"}
print(my_set)
print(len(my_set))
print(type(my_set))

my_set.add ("vacanta")
print(my_set)

#Cum eliminam date din set
#Pentru a elimina un element din set folosim ".remove"

my_set= {"apa","paie", "bataie"}
my_set.remove("bataie")
print(my_set)

#Cum inlocuim date din set
# Pentru a inlocui date din set trebuie mai intai sa eliminam ce dorim sa inlocuim si apoi sa adaugam, asta pentru ca seturile de date nu suporta update de date direct ( cu operatorul =
#ca si in cazul listelor )

my_set= {"acasa","mergem","amandoi", "beti"}
print(my_set)

my_set.remove("beti")
my_set.add("fericiti")

print(my_set)

#Cum facem update la un set cu elemente multiple
#Pentru a face update cu elemente multiple folosim ".update" ( cu metoda update putem adauga la set o lista de exemplu )

my_set = {1,2,3}
my_set.update([4,5,6])
print(my_set)

# Seturile nu pot contine elemente mutabile ca si dictionarele, prin urmare aceasta functie nu se aplica in Python ( exemplul de mai sus unde am avut mai multe dictionare integrate in lista )

#Tuple
#Accesarea unei valori in Tuple se face prin index ( la fel ca si in cazul listelor )
my_tuple=("apple", "banana","cherry")
first_item= my_tuple[0]
print(first_item)

# Pentru a face update in tuple
# Dupa cum stim, tuple nu poate fi modificat o data ce a fost definit iar pentru a face update trebuie sa cream un nou tuple:

#1. Dupa ce am creat tuple ( my_tuple=("apple", "banana", "chery") ) putem lua primele 2 elemente prin comma notation: my_tuple[:2] adica toate elementele de la inceputul de tuple pana la index 2
#dar mai mici de 2.
#2. Urmatorul pas e sa cream un nou tuple pentru a face update : my_tuple= my_tuple[:2]+("orange)

my_tuple=my_tuple[:2]+("orange",) # necesar sa punem virgula pentru a integra in tupple ( cel putin asa o inteleg eu )
print(my_tuple)

#Accesarea de Keys a Dictionarelor integrate in Tuple
#Daca un tuple contine un dictionar, trebuie sa accesam mai intai dictionarul prin indexul lui apoi sa accesam key in dictionar

my_tuple=(
    {"name":"Ionut","age": 34},
    {"name":"Moraru","age":35}
)
name_of_first_person=my_tuple[0]["name"]
print(name_of_first_person)

# Cum printam age ?
age_of_the_first_person= my_tuple[0]["age"]
print(age_of_the_first_person)