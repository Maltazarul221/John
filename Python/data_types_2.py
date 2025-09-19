
# Dictionaries
# asocieaza chei unice cu valori corespunzatoare (structura key + value)
# se foloseste { } pentru declarare si ":" pentru separare key/value
# cheile pot fi: int, foat, str, bool, tuple
# valorile pot fi: int, float, str, bool, tuple, list, dict, set, None
person = {
  "name": "Mircea",
  "age": 31
}
print(person["name"])
print(person["age"])

product = {
    "name" : "Laptop",
    "price" : 3500,
    "in_stock" : True
}
print(product)

# List
# sunt create folosind []
# elementele dintr-o lista sunt separate cu ","
# indexul dintr-o lista se numara de la 0 (o lista cu 7 elemente are indexul 0-6)

shopping_list = ["banana","bread","oat milk"]   # o lista cu 3 elemente de tip string
print(shopping_list)
print(shopping_list[0])    # aici luam primul element din lista

numbers = [35,8,12,6]      # o lista cu 4 elemente de tip int
print(numbers)
print(numbers[0])

mixed_list = [35, 'milk', 12, 'sugar', 36.6]   # o lista mixta, intre string-uri si float
print(mixed_list)
print(type(mixed_list))
print(len(mixed_list))

# Sets
# seturile se creaza cu {}
# elementele unui set sunt unice si neordonate ( nu se acceseaza prin index)

fruits = {"apple","banana","cherry"}   # set cu 3 fructe distincte

lst = [8,1,2,2,5.6,3,4,4,5]
set_fara_duplicate = set(lst)    # folosim "set"
print(set_fara_duplicate)

lst_descrescatoare = sorted(set_fara_duplicate, reverse= True)  # o varianta pt sortarea descrescatoare
print(lst_descrescatoare)

# print(set_fara_duplicate[1]) un set nu are indexare (cum are o lista)

numbers= {1,2,2,2,2,3,4,4,5}
print(numbers)


# TUPLE
# se declara folosind ()
# datele sunt ordonate + imutabile (le-am declarat o data si nu mai pot fi modificate)

# a = ("single element")   # nu este sintaxa corecta pentru tuple, va fi de fapt un "string"
one_element_tuple = ("single element",)
print(one_element_tuple)
print(type(one_element_tuple))

fruits_2 = ("apple","banana","cherry")
print(fruits_2[1])
# fruits_2[0] = "strawberry" TypeError (nu putem modifica tuple)

mixed_tuple = (42, 3.14, "apple,", ("banana","cherry"))  # un set care are int + float + string + tuple
print(mixed_tuple[3][0])   # luam elementul de la index 3 si din tuple de la index 0

# exemplu simplu cu o lista ce contine 2 variabile
x = 14
y = "Hello"
lst = [x,y]
print(lst)

# exemplu simplu cu tuple
x = 7
tpl = (1,2,3,x)
print(tpl)
x = 100
tpl = (1,2,3,x)
print(tpl)

# exemplu de un tuple ce contine mai multe tipuri de date
tpl_2 = ("John Doe", 30, ("Python Developer", "Data Analyst"), [3.5, 4.7])
print(tpl_2)
print(type(tpl_2))

people = [
    { "name": "Your name", "age": 19},
    { "name": "Another name", "age":65}
]

first_person = people[0]
print(people[0]["name"])
print(people[1]["age"])

employees = (
  {
    "name": "Alice",
    "role": "Developer",
    "experience": 5
  },
  {
    "name": "Bob",
    "role": "Designer",
    "experience": 3
  }
)
print(employees)

# noua valoare o suprascrie pe cea veche
my_name = "Your Name"
my_name = "Another Name"
print(my_name)

# atribuirea valorii unei variabile in alta variabila
city = "Bucharest"
new_city = city.upper()
print(city)
print(new_city)

# atribuirea valorii unei variabile in alta variabila
my_age = 19
my_old_age = my_age
my_age = 37
print(my_age)
print(my_old_age)

# adunarea intre valori int/float si variabile (care sunt tot int/float)
my_old_age = 18
my_new_age = my_old_age + 23.2
print(my_new_age)
print(type(my_new_age))