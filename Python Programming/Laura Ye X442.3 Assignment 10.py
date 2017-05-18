'''Write a simple Rectangle class. It should do the following:
	• Accepts length and width as parameters when creating a new instance
	• Has a perimeter method that returns the perimeter of the rectangle
	• Has an area method that returns the area of the rectangle
	• Don't worry about coordinates or negative values, etc.
'''

class Rectangle:
    def __init__(self, l, w):
        self.width = w
        self.length = l

    def perimeter(self):
        return self.width * 2 + self.length * 2
    
    def area(self):
        return self.width * self.length



'''Python provides several modules to allow you to easily extend 
some of the basic objects in the language. Among these modules are 
UserDict, UserList, and UserString. (Refer to the chart in 
Topic 10.3 to see what the methods you need to override look like. 
Also, since UserDict and UserList are part of the collection module, 
you can import them using from collections import UserDict and from 
collections import UserList.)
2. Using the UserList module, create a class called Ulist, and 
override the __add__, append, and extend methods so that duplicate 
values will not be added to the list by any of these operations.
'''

from collections import UserList

class Ulist(UserList):
    def __init__(self, initial_list):
        super().__init__()
        for item in initial_list:
            if not self.__isdup(item):
                self.append(item)
            else:
                print("%s is a duplicate, will not be added" % str(item))
        
    def __isdup(self,element):
        '''private method to see if element already exists in Ulist'''
        if element in self:
            return True
        else:
            return False
        
    def __add__(self,anotherlist):
        for item in anotherlist:
            if not self.__isdup(item):
                self.append(item)
                return self
            else:
                print("%s is a duplicate, will not be added" % str(item))
                
    def __iadd__(self,anotherlist):
        for item in anotherlist:
            if not self.__isdup(item):
                self.append(item)
                return self
            else:
                print("%s is a duplicate, will not be added" % str(item))

    def append(self, element):
        if not self.__isdup(element):
            super().append(element)
        else:
            print("%s is a duplicate, will not be added" % str(element))
    
    def extend(self, ext_list):
        for item in ext_list:
            if not self.__isdup(item):
                super().append(item)
            else:
                print("%s is a duplicate, will not be added" % str(item))

