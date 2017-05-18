
# coding: utf-8

# In[ ]:

'''A file with a name like picture.jpg is said to have an extension of "jpg"; 
i.e. the extension of a file is the part of the file after the final period in its name. 
Write a program that takes as an argument the name of a directory (folder) and then finds the extension of each file. 
Then, for each extension found, it prints the number of files with that extension 
and the minimum, average, and maximum size for files with that extension in the selected directory.'''


# In[56]:

import os,re

def extension_stats(directory):
    file_out = {} #dictionary with keys=extension, values are dictionaries with keys=min,max,average,count and values are floats
    size_dict = {} #dictionary with keys=extension, values are lists of file sizes with that extension
    
    filepat = re.compile('.*\.(?P<extension>[-a-zA-Z0-9]*)', re.I)
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            result = filepat.search(f)
            
            if result == None:
                ext = "None"
            else:
                ext = result.group('extension').lower()
            
            size = os.path.getsize(os.path.join(root, os.curdir, f))
            
            #check to see if ext is already in dictionary, if yes, append size to the list
            #otherwise, create a list with one element = size
            if ext in size_dict.keys():
                size_dict[ext].append(size)
            else:
                size_dict[ext]=[size]
                
    #after getting all the sizes from os.walk, process min,max,average, and number of files with that ext
    for key in size_dict.keys():
        stat_dict = {}
        stat_dict["average"] = sum(size_dict[key])/len(size_dict[key])
        stat_dict["min"] = min(size_dict[key])
        stat_dict["max"] = max(size_dict[key])
        stat_dict["count"] = len(size_dict[key])
        
        file_out[key]=stat_dict
    
    return file_out

def ext_stat_print(dict_input):
    #function to print out the results from extension_stats in a prettier way
    
    for key in dict_input.keys():
        print ("Extension statistics for %s is listed below" % key)
        print ("Number of files is %d" % dict_input[key]["count"])
        print ("Minimum file size is %d" % dict_input[key]["min"])
        print ("Average file size is %d" % dict_input[key]["average"])
        print ("Maximum file size is %d" % dict_input[key]["max"])

ext_stat_print(extension_stats("C:\\Users\\lye\\Downloads"))


# In[ ]:



