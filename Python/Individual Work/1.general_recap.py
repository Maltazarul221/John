import sys
print(sys.version)

#Variables
#a variable is created the moment you assign a value to it

x = 5
y = "John"

print(x)
print(y)
print("Hey John!")
print(type("Hey John!"))
print(type(x))
print(type(y))

#they don't need to be declared with any particular type, and you can change them after they have been assigned

x = 4
x = "Sally"

print(x)

#Casting
#if you want to specify the data type of variable , this can be done by casting

x = str(3)
y = int(3)
z = float(3)

print(x)
print(y)
print(z)

#Variable names
#can have a short name ( x, y ) or a more descriptive name ( age, carname , total_value )
#Rules for Python variables:

#A variable name must start with a letter or the underscore character
#A variable name cannot start with a number
#A variable name can only contain alphanumeric characters and underscores (A-z, 0-9, and _ )
#Variable names are case-sensitive (age, Age and AGE are three different variables)
#A variable name cannot be any of the Python keywords.

# Examples of variables:

myvar = "John"
my_var = "John"
_my_var = "John"
myVar = "John"
MYVAR = "John"
myvar2 = "John"

# Asign multiple values to variables:
# Python allows us to asign values to multiple variables in one line

x , y , z , = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)

# One value to multiple variables:

x = y = z = "Orange"

print(x)
print(y)
print(z)

#Unpack a collection
#If you have a collection of values in a list, tuple etc. Python allows you to extract the values into variables.
# This is called unpacking.

fruits = ["Apple", "Banana", "Cherry"]
x,y,z = fruits

print(x)
print(y)
print(z)

# Output variables:
# In python print() is often used to output variables:

x = "Python is awesome"
print(x)

# we can output multiple variables separeted by comma:

x = "Python"
y = "is"
z = "awesome"
print(x,y,z)

#we cal also use the operator + for the same@

x = "Python"
y = " is"
z = " awesome"
print(x + y + z)

#Global variables
#that are created outside a function (as in all the examples in the previous pages) are known as global variables.
#Global variables can be used by everyone, both inside of functions and outside.

x = "Awesome"

def myfunc():
    print("Python is " + x)

myfunc()

#If you create a variable with the same name inside a function, this variable will be local,
# and can only be used inside the function.
# The global variable with the same name will remain as it was, global and with the original value.

x = "Awesome"

def myfunc():
    x = "Fantastic"
    print("Python is " + x)

myfunc()
print("Python is " + x)

# Global Keyword
# Normally, when you create a variable inside a function, that variable is local,
# and can only be used inside that function.
# To create a global variable inside a function, you can use the global keyword.
# Example:

def myfunc():
    global x
    x = "Fantastic!"

myfunc()
print("Python is " + x)

#Also, use the global keyword if you want to change a global variable inside a function.
#To change the value of a global variable inside a function, refer to the variable by using the global keyword:

x = "Awesome!!"

def myfunc():
    global x
    x = "Double Awesome!!!"

myfunc()

print("Python is " + x)

#Python data types#
#Build in data types:

#Text type: str
x = "Hello World"
print(x)

#Numeric type: int
x = 20
print(x)

#Numeric type: float
x = 21.5
print(x)

#Numeric type : complex
x = 1j
print(x)

#Sequence types: list
x = ["Apple", "Banana", "Cherry"]
print(x)

#Sequence types: tuple
x = ("Apple", "Banana", "Cherry")
print(x)

#Sequence types: range
x = range(6)
print(x)

#Mapping type: dic
x = {"Name": "John", "Age": 34}
print(x)

#set type: set
x = {"Apple", "Banana", "Cherry"}
print(x)

#set type:frozenset
x = frozenset({"apple", "banana", "cherry"})
print(x)

#Boolean type: bool
x = True
print(x)

#Binary type: bytes
x = b"Hello"
print(x)

#Binary type: bytearray
x = bytearray(5)
print(x)

#Binary type: memory
x = memoryview(bytes(5))
print(x)

#None type: NoneType
x = None
print(x)

#Setting the specific data types:

x = str("Hello World")
print(x)
x = int(20)
print(x)
x = float(20.5)
print(x)
x = complex(1j)
print(x)
x = list(("apple", "banana", "cherry"))
print(x)
x = tuple(("apple", "banana", "cherry"))
print(x)
x = range(18)
print(x)
x = dict(name="John", age=36)
print(x)
x = {"apple", "banana", "cherry"}
print(x)
x = frozenset(("apple", "banana", "cherry"))
print(x)
x = bool(5)
print(x)
x = bytes(10)
print(x)
x = bytearray(12)
print(x)
x = memoryview(bytes(5))

#Python Numbers:
#There are 3 numeric types in Python:

a = 1 # int
b = 2.8 # float
c = 1j # complex

#Int
#Int, or integer, is a whole number, positive or negative, without decimals, of unlimited length.

d = 1
e = 3474827471246
f = -2134144

print(type(d))
print(type(e))
print(type(f))

#Float
#Float, or "floating point number" is a number, positive or negative, containing one or more decimals.

m = 1.10
n = 1.1
o = -25.32

print(type(m))
print(type(n))
print(type(o))

#Float can also be scientific numbers with an "e" to indicate the power of 10.

x = 35e4
y = 12E4
Z = -87.7e100

print(type(x))
print(type(y))
print(type(z))

#Complex
#Complex numbers are written with a "j" as the imaginary part:

x = 3+5j
y = 5j
z = -5j

print(type(x))
print(type(y))
print(type(z))

#Type Conversion
#You can convert from one type to another with the int(), float(), and complex() methods:

x = 1    # int
y = 2.8  # float
z = 1j   # complex

#convert from int to float:
a = float(x)

#convert from float to int:
b = int(y)

#convert from int to complex:
c = complex(x)

print(a)
print(b)
print(c)

print(type(a))
print(type(b))
print(type(c))

#Random Number@
#Python does not have a random() function to make a random number,
# but Python has a built-in module called random that can be used to make random numbers:

import random
print(random.randrange(1 , 15))

z = 5
z = float(z)
print(z)

#Python Casting

#Specify a Variable Type
#There may be times when you want to specify a type on to a variable. This can be done with casting.
# Python is an object-orientated language, and as such it uses classes to define data types, including its primitive types.
#Casting in python is therefore done using constructor functions:
#int() - constructs an integer number from an integer literal, a float literal (by removing all decimals),
#or a string literal (providing the string represents a whole number)
#float() - constructs a float number from an integer literal,
# a float literal or a string literal (providing the string represents a float or an integer)
#str() - constructs a string from a wide variety of data types, including strings, integer literals and float literals

#Int Casting

x = int(1)
y = int(2.8)
z = int("3")
print(x)
print(y)
print(z)

#Float Casting

x = float(1)
y = float(2.8)
z = float("3")
w = float("4.2")
print(x,y,z,w)

#Strings Casting

x = str("s1")
y = str(2)
z = str(3.0)
print(z,y,z)

#Python Strings
#Strings in python are surrounded by either single quotation marks, or double quotation marks.
#'hello' is the same as "hello".
#You can display a string literal with the print() function:

print("Hello")
print('Hello')

#Quotes insite quotes:

print("It's alright")
print("He is called 'Johny'")
print('He is called "Johny"')

#Assign string to a variable

a = "Hello"
print(a)

#Multiple Strings
#We can use three double quotes:

a = """this is just  text
to prove the 3 double quotes"""

print(a)

#Strings are arrays
#Like many other popular programming languages, strings in Python are arrays of unicode characters.
#However, Python does not have a character data type, a single character is simply a string with a length of 1.
#Square brackets can be used to access elements of the string.
#Example:

a = "Hello World"
print(a[1])

# Looping through a string
#Since strings are arrays, we can loop through the characters in a string, with a for loop.

for x in "banana":
    print(x)

#String Length

a = "Hello, World!"
print(len(a))

#Check String

#To check if a certain phrase or character is present in a string, we can use the keyword in.

txt = "The best things in life are free!"
print("free" in txt)