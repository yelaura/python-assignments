
# coding: utf-8

#Using the keys method for dictionaries and the sort method for lists, write a for loop that prints the keys and corresponding values of a dictionary in the alphabetical order of the keys.

# In[1]:

new_dict = {"a":"apple", "b":"boy", "d":"dancing", "c":"cars"}
dict_keys = list(new_dict.keys())
dict_keys.sort()
for item in dict_keys:
    print ("Key is " + str(item) + " and value is " + str(new_dict[item]) +"\n")


#As an alternative to the range function, some programmers like to increment a counter inside a while loop and stop the while loop when the counter is no longer less than the length of the array. Rewrite the following code using a while loop instead of a for loop.

# In[2]:

a = [7,12,9,14,15,18,12] 
b = [9,14,8,3,15,17,15] 
big = [ ] 
n = 0
while n < len(a):
    big.append (max(a[n],b[n])) 
    n += 1
print (big)


#Write a loop that reads each line of a file. It should count the number of characters in each line and keep track of the total number of characters read. Once you have a total of 1,000 or more characters, break from the loop. (You can use a break statement to do this.)

# In[3]:

file = open("untitled.txt", 'r')
lines = file.readlines()
counter = 0
for item in lines:
    counter += len(item)
    print ("Character count is " + str(counter))
    if counter > 1000:
        print ("There are more than 1000 characters in this file")
        break
else:
    print ("There are less than 1000 characters in this file")


#Modify the program written in question 3 so that it doesn't count characters on any line that begins with a pound sign (#).	
	
# In[4]:

file = open("untitled.txt", 'r')
lines = file.readlines()
counter = 0
for item in lines:
    if item[0] is not "#":
        counter += len(item)
        print ("Character count is " + str(counter))
    else:
        print ("A line has been commented out")
    
    if counter >= 1000:
        print ("There are more than 1000 characters in this file")
        break
else:
    print ("There are less than 1000 characters in this file")
