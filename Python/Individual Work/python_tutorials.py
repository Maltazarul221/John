from math import *

#Variables and data types
#Usign variables in Python is useful as we are going to deal with a lot of data, and in the program we are going to use a lot of data, values and information,
#and sometimes this data is difficult to manage, and that's why we have variables which is a container were we can store certain data values.

character_name= "Tom" #this is a string
character_age= "50" #numbers can be stored as strings as well
is_male= True #booleans can be stored as strings

print( "There once was a man named " + character_name+",")
print( "he was "  +character_age+  " years old.")

character_name= "Mike" #this will change the character name

print("He really liked the name " + character_name+",")
print("But he did not liked being " + character_age+ ".")

print(is_male)

#In the example above, we see how variables can be handy if we want to manage and update data easily.
#In Python, there are 3 types of data that we work with in a majority of cases:
#1- STRINGS
#2- NUMBERS (WHOLE OR DECIMAL NUMBERS)
#3- TRUE OR FALSE VALUES (BOOLEANS)

# Working with Strings

# In Python, the most common type of data that we are going to use is strings that are simple plain text that we want to have in our program.

# We can do several things with the string through different commands

print("Trebuie sa\n invatam Python") # \n we can use this to print our string in 2 lines
print("Trebuie sa \" invatam Python") # \" this is going to print quotation mark
print("Trebuie sa \ invatam Python") # this will simply print the backslash

#To simplify, we can create a string variable to use on print

phrase="Trebuie sa invatam Python"
print(phrase)

# We can use concatenation (this is the process of taking a string to add it to another string )

print(phrase+"in maxim 2 saptamani:D")

#Diffrent function in strings

print(phrase.lower()) # we can print only in lower case
print(phrase.upper()) # we can print only in upper case
print(phrase.islower()) # we can check if its lower (prints false)
print(phrase.isupper()) # we can check if its upper (prints false)
print(phrase.upper().isupper()) # phrase.upper has changed the text in upper keys and now will print True
print(phrase.lower().islower()) # same for lower case and it will print true
print(len(phrase)) # this will show us the length of the phrase in the string
print(phrase[5]) # we can use this to print just a letter from the phrase, [5] is the index of letter i
print(phrase[-5]) # this will print the letter from the end of the phrase as [-5] will count backwords
print(phrase.index("s")) # using this function will show us the index where the letter "s" is in the phrase
print(phrase.replace("invatam", "repetam")) # this will replace the word "invatam" with "repetam"

#Working with numbers

print(1) #this is just going to print a number
print(-547) #this is just going to print a negative number
print(5+6) #this will add the number together and will print the total
print(12*5*5+2) #this will print the total in order of operations
print(3*(5.5+5))
print(10%5) # remainder

my_num= 25
print(my_num)
print(str(my_num)) #this will create a string with our number and will be handy when we want to print numbers along with strings
print(str(my_num)+ " Is not my favourite number ") #This, for example

#Dffrent Functions we can use with numbers

my_number= -125
print(abs(my_number)) # this will print the absolute number
print(pow(3,2)) # this will raise 3 to the power of 2
print(max(4,6)) # print the max between 2 numbers
print(min(25,2648)) #print the min between 2 numbers
print(round(12.3))# this will round the number to the closest value
print(floor(4.5)) # this will grab and print the lowest number as it's going to remove the decimal point
print(ceil(3.1)) # this will round up the number no matter what
print(sqrt(36))



