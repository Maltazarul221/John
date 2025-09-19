
my_personal_data = {
    "name": "Your Name",
    "age": 19
}
# daca folosesc "get" si cheia nu exista, imi returneaza None
print(my_personal_data.get("namesdsds"))
print(my_personal_data.get("age"))

# daca vreau sa accesez o cheie care nu exista, arucan eroarea "KeyError"
print(my_personal_data["name"])
print(my_personal_data["age"])

# update al unei valori in Dictionary
my_personal_data["age"] = 20
my_new_age = my_personal_data["age"]  # 20
print(my_new_age)

# accesarea cheilor
my_people_list = [
  {
    "name": "Your Name",
    "age": 19
  },
  {
    "name": "Another Name",
    "age": 65
  }
]

# a citi o valoarea a unei chei dintr-un dict, aflat intr-o lista
my_first_person_name = my_people_list[0]["name"]

# update al unei valori dintr-un dict, care se afla intr-o lista
my_people_list[0]["name"] = "New Name"
print(my_people_list)


# accesarea elementelor intr-o lista
my_shopping_list = ["banana", "bread", "oat milk"]

my_first_item = my_shopping_list[0]  # "banana"
my_second_item = my_shopping_list[1]  # "bread"
last_item = my_shopping_list[-1]     # indexi negativi incep de la "-1", la "-1" e ultimul elemente
print(last_item)

my_shopping_list[0] = "olive oil"   # update al unei valori intr-o lista
print(my_shopping_list)

my_shopping_list.append("milk")     # "append" ne ajuta sa adaugam el intr-o lista
print(my_shopping_list)

my_shopping_list.remove("bread")    # stergem valoarea "bread" din lista
print(my_shopping_list)

my_shopping_list.pop(1)        # sterge ce avem la indexul 1
print(my_shopping_list)



# operatii pe set-uri

fructe = {"mar","banana"}
fructe.add("portocala")   # "add" se foloseste pentru seturi
print(fructe)

my_set = {1, 2, 3}     # definim un set
my_set.add(4)          # adaugam un element in set (se foloseste add la sets, append la liste)
print(my_set)

my_set.remove(3)       # stergem elementul indicat, remove se foloseste atat la liste cat si la sets
print(my_set)

my_set.discard(2)      # o alternativa la "remove", doar ca nu ne da eroare
print(my_set)

my_set_2 = {1, 2, 3}
my_set_2.update([4, 5, 6])    # update() o folosim pt a adauga mai multe elemente dintr-un iterabil
print(my_set_2)


# operatii tuple

my_tuple = ("apple", "banana", "cherry")
first_item = my_tuple[0]
print(first_item)

# my_tuple[:2] = o operatie de slicing (aplicata pe un tuple)
my_tuple = my_tuple[:2] + ("portocala",)
print(my_tuple)

my_tuple = (10,20,30,40,50)
#my_tuple[:2]      # primele 2 elemente (10,20), adica de la inceput (index0) pana la index2 (fara a-l include)
#my_tuple[:4]      # primele 4 elemente (10,20,30,40), adica de la inceput (index0) pana la index4 (fara a-l include)
#my_tuple[1:4]     # (20,30,40) elementele de la index1 la index3 (4-1)
#my_tuple[2:]      # (30,40,50) de la index 2 pana la final