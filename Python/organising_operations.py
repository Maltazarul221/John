
# exercitiu "Organising operations"
# creeaza o var cu o valoare  tip string

this_text = "Hello, This is the text."
def log_into_terminal():
    print(this_text)

log_into_terminal()

# variabile locale
def log_into_terminal():
    local_text = "This a local text."
    print(this_text)
    print(local_text)

# functia "another function" apeleaza "log_into_terminal"
log_into_terminal()
def another_function():
    log_into_terminal()

another_function()

# functia "log_into_terminal" poate fi atribuita unei variabile
third_function = log_into_terminal
third_function()  # executa functia atribuita

# functie in functie

def log_into_terminal():
    local_text = "Local text in principal function."

    def fourth_function():
        print("Hello from fourth function!")
        print(local_text)

    print(this_text)
    print(local_text)
    fourth_function()   # apelam functia interna

# apelam prima functie in cea de a2a
def log_into_terminal_2(message):
    print(f"Mesaj primit: {message}")

def another_function_2():
    log_into_terminal_2("Hello from this function!")

another_function_2()

# functia "greetings" returneaza un mesaj personalizat
def greetings(name):
    return f"Hello,{name}!"

print(greetings("Ana"))
print(greetings("Ion"))
print(greetings("Maria"))
print(greetings("Andreea"))







