import datetime
import time

# explicatie f-string

name = "John"
age = 35
print(f"My name is {name} and I am {age} years old.")


# creata o lista de dictionare cu filmele aferente

favourite_movies = [
    {
        "title": "Inception",
        "year": 2010,
        "rating": 8.8,
        "description": "A nice movie.",
        "directors": ["Christopher Nolan"],
        "writers": ["Christopher Nolan"],
        "actors": ["Leonardo DiCaprio", "Elliot Page"],
        "genres": ["Action","Sci-Fi"]
    },
    {
        "title": "The Matrix",
        "year": 1999,
        "rating": 8.7,
        "description": "An interesting movie.",
        "directors": ["Lana Wachowski"],
        "writers": ["Lana Wachowski"],
        "actors": ["Keanu Reeves", "Elliot Page"],
        "genres": ["Action", "Sci-Fi"]
    },
    {
        "title": "Interstellar",
        "year": 2014,
        "rating": 8.6,
        "description": "A great movie.",
        "directors": ["Lana Wachowski"],
        "writers": ["Lana Wachowski"],
        "actors": ["Mattew McConaughey", "Jessica Chastain"],
        "genres": ["Adventure", "Sci-Fi"]
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "rating": 9.0,
        "description": "An very interesting movie.",
        "directors": ["Lana Wachowski"],
        "writers": ["Lana Wachowski"],
        "actors": ["Christian Bale", "Aaron Eckhart"],
        "genres": ["Drama", "Action","Sci-fi"]
    }
]

# in python, fiecare fisier .py are o variabila speciala numita "__name__"
# daca fisierul e rulat direct (data.py) atunci "__name__" va avea valoarea "__main__"
# daca fisierul e importat in alt fisier .py -> "__name__" va avea valoarea numelui modulului
# codul din blocul if __name__ = __main__ : se executa doar daca fisierul e rulat direct (nu si cand e importat)
if __name__ == "__main__":
    print(favourite_movies)
    # citim anumite informatii din dictionarele aferente filmelor
    print(f"Title of the first movie: {favourite_movies[0]['title']}")
    print(f"Year of the second movie: {favourite_movies[1]['year']}")
    print(f"Rating of the third movie: {favourite_movies[2]['rating']}")
    print(f"Description of the fourth movie: {favourite_movies[3]['description']}")


    print("The lead director of the first movie is: ", favourite_movies[0] ["year"])
    print("The lead director of the first movie is: ")
    print(favourite_movies[0] ["year"])

    # exemplu cu print-uri multiple
    x = 5
    y = 9
    z = "Hello"
    t = 8.65

    print(x,y,z,t)
    print("This is a message",t,y,x)

    # citim anumite informatii din dictionarele aferente filmelor
    print(f"The lead director of the first movie is: {favourite_movies[0]['directors']}")
    print(f"The lead writer of the second movie is: {favourite_movies[1]['writers']}")
    print(f"The lead star of the movie is: {favourite_movies[2]['actors'][0]}")
    print(f"The main genre of the fourth movie is: {favourite_movies[3]['genres'][0]}")

    # media aritmetica a rating-urilor = (8.8 + 9.0 + 8.7 + 8.6) / 4
    average_rating = (favourite_movies[0]['rating'] +
                      favourite_movies[1]['rating'] +
                      favourite_movies[2]['rating'] +
                      favourite_movies[3]['rating']
                      ) / 4
    print(average_rating)

    current_year = datetime.datetime.now().year   # folosim structura "modul.functie"
    print(current_year)
    average_age = sum([4 * current_year - favourite_movies[0]["year"] - favourite_movies[1]["year"] - favourite_movies[2]["year"] - favourite_movies[3]["year"] ])/4
    print(average_age)

    # un exemplu cu modulul "time"
    print("Acesta este primul string")
    time.sleep(3)   # "sleep" 3 secunde
    print("Acesta este al doilea string")

    # "dir" si "help" le folosim pentru a afla mai multe despre un modul/functie din python
    print(dir(datetime))
    help(time)
