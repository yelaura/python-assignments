
# coding: utf-8

# In[93]:

#Part 1 - create your data:
#1. Include a section with your name
name = 'Laura Ye'

#2. work only with these imports:
from numpy import matrix, array, random, min, max
import pylab as plb

#3. Create a list A of 600 random numbers bound between (0:10)
A = list(random.randint(0,10,600))

#4. Create an array B with 500 elements bound in the range [-3*pi:2*pi]
B = plb.linspace(-3*plb.pi, 2*plb.pi, 500)

#5. Using if, for or while, create a function that overwrites every element in A that falls outside of the interval [2:9), and overwrite that element with the average between the smallest and largest element in A
for i in range(0,len(A)):
    if A[i] >= 9 or A[i] < 2:
        A[i] = (max(A) + min(A))/2

#6. Normalize each list element to be bound between [0:0.1]
def normalize(L):
    factor = array([0.1/max(L)]*len(L))
    return list(array(L)*factor)

#7. Return the result from the function to C
C = normalize(A)

#8. Cast C as an array
C = array(C)

#9. Add C to B (think of C as noise) and record the result in D
D = C[:len(B)]+B

#Part 2 - plotting:
#10. Create a figure, give it a title and specify your own size and pdi
plb.figure(figsize=(10,6), dpi=120)
fig = plb.gcf()
fig.canvas.set_window_title("Plots")

#11.Plot the sin of D, in the (2,1,1) location of the figure
plb.subplot(2,1,1)
plb.plot(D, plb.sin(D), color='blue', linewidth=5, linestyle='-')

#12. Overlay a plot of cos using D, with different color, thickness and type of line.
plb.hold(True)
plb.plot(D, plb.cos(D), color='red', linewidth=2, linestyle='--')

#13. Create some space on top and bottom of the plot (on the y axis) and show the grid
plb.ylim(min(plb.sin(D)) - 2, max(plb.sin(D)) + 2)
plb.grid()

#14. Specify the following: title, y-axis label and legend to fit in the best way
plb.title('Sin, Cos and Tan of D')
plb.legend(['sin','cos'], loc='lower right')
plb.ylabel('amplitude')

#15. Plot the tan of D, in location (2,1,2) with grid showing, x-axis label, y-axis label and legend on top right
plb.subplot(2,1,2)
plb.plot(D, plb.tan(D))
plb.legend(['tan'], loc='upper right')
plb.grid()
plb.xlabel('period')
plb.ylabel('amplitude')

plb.show()
plb.close()

#16. Organize your code: use each line from this HW as a comment line before coding each step
#17. Save thesesteps in a .py file and email to me before next class. I will run it!


# In[ ]:



