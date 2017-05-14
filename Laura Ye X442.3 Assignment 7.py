
# coding: utf-8

# In[ ]:

import string

'''
Suppose you want to determine whether an arbitrary text string can be converted to a number. Write a function that uses a try/except clause to solve this problem. Can you think of another way to solve this problem?
'''
def str_to_int(strInput):
    try:
        intOutput = float(strInput)
        return intOutput
    except ValueError as errormsg:
        print("Input is not a convertible string")
        return None

'''
Another way to solve this problem would be to iterate through the string input.
If every character is part of the built-in string.digits, then it's convertible
If there is one character that is not part of the built-in string.digits or a decimal point), then it's not convertible
'''
def str_to_int1(strInput):
    digits = string.digits + "."
    for i in strInput:
        if i not in digits:
            print ("Input is not a convertible string")
            return None
    
    return float(strInput)


# In[ ]:

'''
The input function will read a single line of text from the terminal. If you wanted to read several lines of text, you could embed this function inside a while loop and terminate the loop when the user of the program presses the interrupt key (Ctrl-C under UNIX, Linux and Windows.) Write such a program, and note its behavior when it receives the interrupt signal. Use a try/except clause to make the program behave more gracefully.
'''
import sys

def inputText():
    while 1:
        try:
            text = input("Some text you want: ")
            print ("\nWhat I got: ", text)
        except KeyboardInterrupt:
            print ("\nOkay we're done here.")
            sys.exit()

inputText()

