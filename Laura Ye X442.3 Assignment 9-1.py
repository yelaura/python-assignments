'''1. Using os.walk, write a script that will print the filenames of zero length files. 
It should also print the count of zero length files.'''

import os

count = 0
filenames = []

for root, dirs, files in os.walk(os.pardir):
    for f in files:
        if os.path.getsize(os.path.join(root, os.curdir, f)) == 0:
            print(str(f))
            count += 1
            
print ("There are %d zero length file(s)" % count)

'''2. Write a script that will list and count all of the images in a given HTML web page/file. 
You can assume that:
• Each image file is enclosed with the tag <img and ends with >
• The HTML page/file is syntactically correct
'''

import re, urllib.request

imagepat = re.compile('<img.*?src="?(.*?)"? .*?>',re.I)

def get_image_list(url):
    try:
        f = urllib.request.urlopen(url)
    except IOError:
        sys.stderr.write("Couldn't connect to %s" % url)
        sys.exit(1)
    contents = str(f.read())
    f.close()
    
    images = imagepat.findall(contents)
    
    print ("There are %d image(s)" % len(images))
    
    for item in images:
        print (str(item))

get_image_list("http://csb.stanford.edu/class/public/pages/sykes_webdesign/05_simple.html")
get_image_list("http://convertcase.net")
get_image_list("http://www.facebook.com")
