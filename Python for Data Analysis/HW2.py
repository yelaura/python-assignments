
# coding: utf-8

# In[70]:

import numpy as np

#1. Include a section with your name
name = "Laura Ye"

#2. Create matrix A with size (3,5) containing random numbers
A = np.random.randint(0,10,15).reshape(3,5)
print("A is:")
print(A)

#3. Find the size and length of matrix A
print("Size of A is " + str(A.size) + " and length of A is " + str(A.itemsize))

#4. Resize (crop) matrix A to size (3,4)
A = A[:, 0:4]
print("\nCropped A looks like this:")
print (A)

#5. Find the transpose of matrix A and assign it to B
B = A.T
print("\nThis is B:")
print(B)

#6. Find the minimum value in column 1 of matrix B
print("\nMinimum value in column 1 of matrix B is: " + str(B[:,1].min()))

#7. Find the minimum and maximum values for the entire matrix A
print("Minimum value in matrix A is: " + str(A.min()))
print("Maximum value in matrix B is: " + str(A.max()))

#8. Create a Vector X (an array) with 4 random numbers
X = np.random.randint(0,10,4)
print("\nThis is X:")
print(X)

#9. Create a function and pass Vector X and matrix A in it.
#10. In the new function multiply Vector X with matrix A and assign the result to D
def foo(arr1,arr2):
    return arr1*arr2
D = foo(X,A)
print("\nThis is D:")
print(D)

#11. Create a complex number Z with absolute and real parts != 0
Z = 3 + 4j
print("\nThis is Z: " + str(Z))

#12. Show its real and imaginary parts as well as its absolute value
print("real part of Z is " + str(Z.real))
print("imaginary part of Z is " + str(Z.imag))
print("absolute value of Z is " + str(abs(Z)))

#13. Multiple result D with the absolute value of Z and record it to C
C = D * abs(Z)
print("\nThis is C:")
print(C)

#14. Convert matrix B from a matrix to a string and overwrite B
B = B.tostring()
print("\nThis is B as a string:")
print(B)

#15. Display a text on the screen: 'Name is done with HW2' but pass your 'Name' as a string variable
print("\n" + name + " is done with HW2")

#16. Organize your code: use each line from this assignment as a comment line before each step.
#17. Save all steps as a script in .py file.
#18. Email me your .py file and screenshots of your running code before next class. I will run it

