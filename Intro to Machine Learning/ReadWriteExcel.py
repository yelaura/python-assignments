
# coding: utf-8

# # Import packages

# In[2]:

import numpy as np


# In[3]:

import matplotlib.pyplot as plt


# # Function Definitions

# In[1]:

#This function is used in the function readExcel(...) defined further below
def readExcelSheet1(excelfile):
    from pandas import read_excel
    return (read_excel(excelfile)).values


# In[4]:

#This function is used in the function readExcel(...) defined further below
def readExcelRange(excelfile,sheetname="Sheet1",startrow=1,endrow=1,startcol=1,endcol=1):
    from pandas import read_excel
    values=(read_excel(excelfile, sheetname,header=None)).values;
    return values[startrow-1:endrow,startcol-1:endcol]


# In[5]:

#This is the function you can actually use within your program.
#See manner of usage further below in the section "Prepare Data"
def readExcel(excelfile,**args):
    if args:
        data=readExcelRange(excelfile,**args)
    else:
        data=readExcelSheet1(excelfile)
    if data.shape==(1,1):
        return data[0,0]
    elif (data.shape)[0]==1:
        return data[0]
    else:
        return data


# In[6]:

def writeExcelData(x,excelfile,sheetname,startrow,startcol):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df=DataFrame(x)
    book = load_workbook(excelfile)
    writer = ExcelWriter(excelfile, engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheetname,startrow=startrow-1, startcol=startcol-1, header=False, index=False)
    writer.save()
    writer.close()


# In[7]:

def getSheetNames(excelfile):
    from pandas import ExcelFile
    return (ExcelFile(excelfile)).sheet_names


# In[1]:

def __init__():
    pass


# In[ ]:



