from operator import truediv

from data_types_2 import set_fara_duplicate, one_element_tuple

text1= 'this is a single quoted string'
text2= "this is a double quoted string"
text3= "this text dose not represent the weather today :))"
print(text1)
print(type(text1))

var = "12584.3"
text4= "this"+"that"
print(text4)
text5="text4"+"text1"
print(text5)
text6 = text1 + "" + text3

i_have_worked_today = True
you_have_been_off_today = True
the_weather_has_been_nice_today = False

print(i_have_worked_today)
print(you_have_been_off_today)
print(the_weather_has_been_nice_today)
print(type(you_have_been_off_today))
print(bin(145785))
print(abs(-5486544))
print(hex(45844541474755))

person = {
    "name":"John Doe",
    "age": "99",
    "city": "Maybe MOON",
    "marital status":"not known"
}
print(person)
print(type(person))

product = {
    "name": "slippers",
    "type": "footwear",
    "color": "black",
    "in stock": "0"
}
print(product)
print(type(product))

shopping_list = ["milk", "shoes", "apples", "sugar"]
print(shopping_list)
print(type(shopping_list))
print(shopping_list[2])
print(shopping_list[1])
print(shopping_list[3])

fruits = {"cars","boots","shirts","nails"}
print(fruits)
print(type(fruits))

lst = [12,12,12,12,12,21,23,23,24,2,5,25,25,28,20,27]
set_fara_duplicate= set(lst)
print(set_fara_duplicate)
lst_descr = sorted(set_fara_duplicate, reverse= True)
print(lst_descr)
print(type(lst_descr))

lst = ["car","cat","car","car","car","car","dog","frog"]
print(lst)
set_fara_duplicate= set(lst)
print(set_fara_duplicate)
lst_descr= sorted(set_fara_duplicate, reverse= True)
print(lst_descr)

one_element_tuple = ("single element",)
print(one_element_tuple)
print(type(one_element_tuple))
fruits_2 = ("apple", "banana", "carrots")
print(fruits_2[2])
mixed_set = (42, 3.14, "apple,",("apple", "banana", "carrots"))

print(mixed_set)
print(type(mixed_set))
print(type(one_element_tuple))
print(type(lst))
print(type(fruits))
print(type(person))

lst = [ "car"]
print(lst)
print(type(lst))

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
print(type(employees))

my_name: str = "Ion Moraru"
my_age = "22"

my_personal_data= {"name":"Ion Moraru","age":22}

print(my_personal_data)
print(type(my_personal_data))

my_first_five_things = ["Apples", "cars", "smoking","sleeping", "eating"]
print(my_first_five_things)
print(type(my_first_five_things))

my_best_friends = [
    {"name":"Mariana", "age": 45},
    {"name":"Nelu", "age":46}
]
print(my_best_friends)
print(type(my_best_friends))

my_name = "Ion"
my_name = "Maltazarul"

my_name = "Moraru"
my_new_name= my_name

print(my_name)
print(my_new_name)

my_age= 56
my_new_age= my_age
var = my_age - 22

print(my_age)

my_old_age= 55
my_new_age= my_old_age + 8
print(my_new_age)

my_name = "NELU"
print(my_name)
my_age= "46"
print(my_age)

my_personal_data= {"name": "Nelu", "age":"46"}
print(my_personal_data)

my_first_three_things= ["ai","wine","coffee"]
print(my_first_three_things)

my_best_friends = [
    {"name":"Marius","age":45},
    {"name":"Denisa","age":55}
]
print(my_best_friends)

my_name="cheregi"
my_name="cheregi"

my_name="nelu"
my_new_name=my_name

print(my_name)

my_age=17
my_new_age = my_age
my_age=27

print(my_new_age)

my_old_age = 777
my_new_age= my_old_age + 125

print(my_new_age)
print(type(my_new_age))

my_personal_data = {"name": "Ion Moraru", "age": 20}

my_new_age=my_personal_data["age"]
my_personal_data.get("name")
my_personal_data.get("age")
print(my_personal_data)
print(type(my_personal_data))

my_personal_data ["name"]
var = my_personal_data["age"]

print(my_personal_data)
print(type(my_personal_data))

print(my_new_age)