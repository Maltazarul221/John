"""
Acest modul contine practici bune de scriere a codului.
Autor: Mircea Neagu
Data: 01-09-2025
"""

# variabilele trebuie sa contina litere mici, daca avem mai multe cuvinte, separate cu "_"
student_count = 15
total_number_of_products = 56
# de evitat nume vagi, generice sau abrevieri excesive
# ex de evitat: a,b,c,var,x,data


# constantele se scriu conventional cu majuscule pentru a le distinge de variabile
# ex: MAX_SIZE, DEFAULT_COLOR
MAX_SIZE = 100
DEFAULT_COLOR = "red"
# daca vedeti ceva de genul, ele sunt considerate constante (prin conventie) si nu ar trebui modificate
# poate fi si o lista considerata constanta


# functii - e necesar sa folosim litere mici separate prin "_"
# ex: calculate_total, print_report
# de evitat nume vagi, de ex: func, do_something, create_something


# Clase - o clasa este un sablon pentru crearea obiectelor.
# numele claselor trebuie sa inceapa cu litera mare, daca sunt 2 cuvinte, fiecare cuvant incepe cu litera mare
# ex: StudentProfile, InventoryManager
# obiectele trebuie trebuie cu litera mica
# student_1 = StudentProfile() - obiectul "student_1" e instanta clasei StudentProfile()

# foloseste 4 spatii pentru indentare (standard Python)
# nu supra-comenta cod simplu si nu scrie comentarii redundante



# docstring-uri
# se scriu intre triple quotes """   """
# servesc pentru documentarea functiilor, claselor sau modulelor (fisierele .py)
# sunt accesibile la runtine, de ex cu help()

def calculate_area(raza):
    """
    Calculeaza aria unui cerc
    :param raza: raza cercului
    :return: aria cercului
    """
    return 3.14159 * raza ** 2

help(calculate_area)

# foloseste spatiu in jurul operatorilor si dupa virgule
# se aplica aceasta practica si pentru liste/tuple/set si pentru argumentele unei functii
items = [1, 2, 3, 4, 5]
age = 18
total = age + 5

# foloseste conditii clare si explicite la structuri "if"
# ex: if age >=18:
# evita conditii complicate sau if-uri foarte impricate
# evita: if(age >=18 and age <65) or (membership and not banned) and (score > 100)
# codul e greu de citit si de intretinut
# trebuie sa analizezi fiecare parte ca sa intelegi logica + creste riscul de erori


# foloseste "list comprehension" pentru transformari simple de date
# ex: [x ** 2 for x in numbers]
# evita comprehensions complexe sau imricate (dificil de citit)


# foloseste try-except pt erori asteptate
# Nu captura toate exceptiile, fii specific cu tipurile de exceptii


# la crearea de functii, da nume foarte sugestive:
# exemplu corect mai jos:
def calculate_discount(price, discount):
    final_price = price - (price * discount / 100)
    return final_price

discounted_price = calculate_discount(100, 10)

# de evitat, ceva de genul (unde nu avem nume specifice):
# Define function
def calc_d(p, d):
    fp = p - (p * d / 100)  # Calculate result
    return fp  # Return result

# Unclear and vague
dp = calc_d(100, 10)  # Call function with the following arguments: 100 and 10

# __name__ e o var interna pe care Python o seteaza automat
# daca rulezi un fisier direct (good_practice.py) atunci __name__ == __main__

def add(a, b):
    return a + b

def main():
    print("Test add:", add(2, 3))

# daca fisierul e rulat direct, codul de bloc se va executa
# protejeaza codul de test sau codul executabil
if __name__ == "__main__":
    main()








