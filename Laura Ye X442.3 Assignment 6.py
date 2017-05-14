
# coding: utf-8

'''
Write a function that accepts the name of a file and returns a tuple containing the number of lines, words, and characters that are in the file. (Hint: Use the split function of the string module to help you count the words in the file.) 
'''
def lines_words_char(filename):
    fileobject = open(filename)
    list_of_lines=fileobject.readlines()
    list_of_lines=[x[:-1] for x in list_of_lines]

    list_of_words=[]
    for item in list_of_lines:
        list_of_words.extend(item.split(" "))
    list_of_words=[x for x in list_of_words if x != '']
    
    list_of_chars=""
    for item in list_of_words:
        list_of_chars = list_of_chars + item
    
    return (len(list_of_lines), len(list_of_words), len(list_of_chars))

'''
Write a function that accepts an arbitrary number of lists and returns a single list with exactly one occurrence of each element that appears in any of the input lists. 
'''
def merge_lists(*lists):
    final_list = lists[0]
    
    for list_item in lists[1:]:
        for item in list_item:
            if item not in final_list:
                final_list.append(item)
    
    return final_list

'''
Use the map function to add a constant to each element of a list. Perform the same operation using a list comprehension. 
For example, the list (1, 20, 300, 400) and constant 8, will result in: 9, 28, 308, 408
'''
a = list(range(0,15))
a_listcomp = [x+10 for x in a]
print (a_listcomp)
a_map = list(map(lambda x: x+10, a))
print (a_map)

'''Write a function that will take a variable number of lists. Each list can contain any number of words. This function should return a dictionary where the words are the keys and the values are the total count of each word in all of the lists

For example, if we are given the following lists:

wl1 = ["double", "triple", "int", "quadruple"]
wl2 = ["double", "home run"]
wl3 = ["int", "double", "float"]

the function should output the following dictionary (The order of the words is not important):
{'float': 1, 'int': 2, 'quadruple': 1, 'home run': 1, 'triple': 1, 'double': 3}

Note, you may have to create an empty dictionary first (for example: dict = {}).
'''
def list_dict(*list_words):
    final_dict = {}
    
    for list_item in list_words:
        for item in list_item:
            if item in final_dict.keys():
                final_dict[item] += 1
            else:
                final_dict[item] = 1
    
    return final_dict

'''
(Optional) Write a function that combines several dictionaries by creating a new dictionary with all the keys of the original ones. If a key appears in more than one input dictionary, the value corresponding to that key in the new dictionary should be a list containing all the values encountered in the input dictionaries that correspond to that key.
Previous Next
'''
def merge_dict(*dicts):
    final_dict = {}
    
    for dict_item in dicts:
        for dict_key in dict_item:
            if dict_key in final_dict.keys():
                if isinstance(final_dict[dict_key], list):
                    final_dict[dict_key].append(dict_item[dict_key])
                else:
                    final_dict[dict_key] = [final_dict[dict_key], dict_item[dict_key]]
            else:
                final_dict[dict_key]=dict_item[dict_key]
    
    return final_dict