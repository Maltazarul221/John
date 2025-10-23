from data import favourite_movies                      # from "fisier.py" import list/dict/funct
from folder_date.data_2 import favourite_movies_2      # from "director.fisier.py" import list/dict/funct
from data import *                                     # from "fisier.py" import * = importi tot fisier.py

# cum importam dintr-un fisier.py care se afla intr-un alt proiect sau oriunde in PC-ul nostru
# sys este un modul built in in Pyhton
# sys.path este o lista in Python care contine toate path-urile in care interpretorul Python cauta module cand faci import
import sys
sys.path.append('C:/Users/User/Desktop/data_science')

# \n ne pune practic o linie goala (empty) in print-ul nostru (daca il folosim la finalul print-ului)
print("Mai jos avem afisarea filmelor folosind o bulca for:\n")
# afisarea filmelor folosind o bucla "for"
for movie in favourite_movies:
    print(movie["title"])

print("Mai jos avem afisarea filmelor folosind o bulca while:")
x = 0      # initializam indexul
while x < len(favourite_movies):
    print(favourite_movies[x]["title"])   # accesam titlul pe baza indexului
    x += 1 # incrementam indexul

# calcularea unei medii a valorii de rating pt fiecare film
total_rate = 0
i = 0       # o vom folosi ca index pt a parcurge lista de filme in bucla while

while i < len(favourite_movies):                  # cautam in "favourite_movies"
    total_rate += favourite_movies[i]["rating"]   # calculam total rate ca suma a rating-urilor
    i += 1

average_rate = total_rate / len(favourite_movies)   # calculam media rating-urilor
print(f"Average rating: {average_rate}")

# identificarea celui mai nou film
newest_movie = favourite_movies[0]     # presupunem ca primul film este cel mai nou
for movie in favourite_movies[1:]:     # incepem bucla de la al doilea film (index 1)
    if movie["year"] > newest_movie["year"]:
        newest_movie = movie              # actualizam variabila


fructe = ["mar","banana","portocala","kiwi"]

# slicing la liste: sintaxa ar fi:  lista[start : stop : step]
print(fructe[1:3])  # de la index 1 la index 3
print(fructe[:3])   # de la inceput pana la index(n - 1), deci index(2) in cazul nostru
print(fructe[2:])   # de la index(n) pana la final
print(fructe[::2])  # ia fiecare al doilea element (deci ia din 2 in 2)

numere = [0,1,2,3,4,5,6,7,8,9]
sublista = numere[1:8:2]  # elementele de la index 1 la index 8-1 , sarind cate 2 elemente
print(sublista)

print(favourite_movies[0]["actors"])
print(favourite_movies[0]["title"])

# listarea actorilor + fiecarui film
stars_by_movies = ""        # initializam un string gol (aici vom aduna toate titlurile si actorii)
for movie in favourite_movies:
    stars_by_movies += movie["title"] + ": "

    for star in movie["actors"]:
        stars_by_movies += star + ", "
                                                  # strip() e o metoda folosita pt string-uri
    stars_by_movies = stars_by_movies.strip(", ") # elimina toate virgulele (,) si spatiile() care apar la inceput sau la sfarsit
    stars_by_movies += "\n"

print(stars_by_movies)

# functie care face exact aceeasi chestie ca mai sus

def list_stars_by_movies(movies):

    stars_by_movies = ""  # initializam un string gol (aici vom aduna toate titlurile si actorii)
    for movie in movies:
        stars_by_movies += movie["title"] + ": "

        for star in movie["actors"]:
            stars_by_movies += star + ", "

        stars_by_movies = stars_by_movies.strip(", ")  # elimina toate virgulele (,) si spatiile() care apar la inceput sau la sfarsit
        stars_by_movies += "\n"

    return stars_by_movies

print(list_stars_by_movies(favourite_movies))
























