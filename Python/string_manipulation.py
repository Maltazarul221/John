# functii de baza pentru string-uri
text = "Hello"
print(len(text))

# convertim din int in string
number = 458
print(str(number))

# metode pentru string-uri (apelate cu .)
# .upper() = toate literele devin mari
text = "Python is a great programming language."
print(text.upper())

# .lower() = toate literele devin mici
text_2 = "Python is a great programming language."
print(text_2.lower())

# .capitalize() = face prima litera mare si restul mici
text_3 = "aceasta este o propozitie."
print(text_3.capitalize())

# cauta un substring in string-ul pe care il dam noi
# .find(substring)
# sintaxa = string.find(substring, start, end) - start/end sunt optionali, substring e obligatoriu!
text_4 = "It's a nice day."
print(text_4.find("nice")) # returneaza 7 (index-ul primului caracter unde incepe substring-ul)
# indexul se ia din string-ul principal
print(text_4.find("hello"))  # daca nu se gaseste substring-ul se returneaza -1
print(text_4.find("nice",9)) # returneaza -1, cautarea incepe de la index 9
print(text_4.find("nice",4,11)) # cauta de la index 4 la index 10
# tot substring-ul pe care il cautam trebuie sa fie intre start si end (nu doar indexul primelei litere)

print("acesta este un text".upper())  # un exemplu cu o functie direct pe string - nu pe o variabila definita anterior
print(text_4[-1].find("nice",-1))

# replace() e o metoda de string care inlocuieste un substring cu un alt substring
# sintaxa = string.replace(old,new,count) - old/new sunt obligatorii, count este optional
text_5 = "We learn python fast. learn"
print(text_5.replace("learn","study",1)) # count = 1 -> inlocuieste doar prima aparitie
print(f"New text: {text_5.replace('learn','study',1)}")

# .startswith(substring)
# verifica daca un string incepe cu un anumit substring
text_6 = "We like python."
print(text_6.startswith("We"))  # returneaza boolean: True daca gaseste, False daca nu

# .endwith(substring)
# verifica daca un string se termina cu un anumit substring
# case-sensitive = se tine cont de litere mari/mici
text_7 = "Python is fun"
print(text_7.endswith("fun")) # True
print(text_7.endswith("Fun")) # false














