
# coding: utf-8

# In[2]:

### Step 0: Import dependencies and define critical functions

# Dependencies for importing data and analysis
import numpy
from scipy import ceil

# Dependencies required for graphing
from IPython import display
import matplotlib.patches as mpatches
import pylab as plb

# Dependency required for extrapolation
from scipy.stats import linregress

## Functions

#function to convert a time string (b"X:XX:XX") to number of seconds
#e.g. "1:02:03" is converted to 3723 (int)
#if data is invalid (e.g. "-") then return -1
def timetosecs(timestring):
    try:
        timestring = numpy.array(timestring.decode('utf-8').split(':'), dtype=int)
        totalsecs = timestring[0] * 60 * 60 + timestring[1] * 60 + timestring[2]
        return totalsecs
    except:
        return -1
    
#Function for removing or filtering data within the array
#If remove is True, then filtered data will NOT contain val
#If remove is False, then filtered data WILL contain val only
#If column = None then all columns will be searched
#If column = int between 0,15 then the specified column will be searched.
def filter_data(arr, val, remove=True,name=None):
    #identifies which indices to REMOVE
    if remove:
        if name==None:
            indices = [i[0] for (i,x) in numpy.ndenumerate(arr) if val in x]
        else:
            indices = [i[0] for (i,x) in numpy.ndenumerate(arr) if val == x[arr.dtype.names.index(name)]]
    else:
        if name==None:
            indices = [i[0] for (i,x) in numpy.ndenumerate(arr) if val not in x]
        else:
            indices = [i[0] for (i,x) in numpy.ndenumerate(arr) if val != x[arr.dtype.names.index(name)]]            
        
    return numpy.delete(arr, indices)


# In[3]:

### Step 1: Load Data from CSV and remove rows with missing values ###

#set up dtypes array as argument for numpy.genfromtxt
dtypes=['i4', '<U5', 'i4', 'U1', 'U2', 'U3']
dtypes.extend(['i4']*9)

#Generate structured array from csv
raw = numpy.genfromtxt('marathon_results_2017.csv', 
                       delimiter=',', 
                       dtype=dtypes, 
                       names=True, #use header as names for each column
                       case_sensitive=False, #case insensitive, changes all column headers to uppercase
                       invalid_raise=False,
                       converters=dict.fromkeys(numpy.arange(6,15,dtype=int), timetosecs) #converts time to number of seconds
                       )

# Removes missing values in running times e.g. "-"
raw_new = filter_data(raw, -1)

#Creating a new array for filtered data
data = raw_new[:][:]


# In[4]:

### Step 2: Filter Data (optional) ###

# User can optionally filter data by names of the array in data
while (1):
    
    userinput = input("Would you like to filter the data? y/n ")
    
    # If user input is "yes" or "y", then prompts user for a category and also value to filter by
    if userinput.lower() == "yes" or userinput.lower() == "y":
        
        # Filter category which are names of the columns in structured array
        # E.g. Age, MF, State, Country, etc
        filtercat = input("What category are you filtering by?\nChoose one of the following:\n"
                          + str(data.dtype.names[2:6])+ "\n").upper()
       
        # Filter value - desired value that the user would like to see
        # E.g. Age = 25, Country = USA, etc
        filterval = input("What values would you like to see? ")
        
        # Conversion of filter value to int or uppercase for input to filter_data function
        if filterval.isnumeric():
            filterval = int(filterval)
        else:
            filterval = filterval.upper()
        
        # Filters data by the desired category and values
        data = filter_data(data, filterval, remove=False, name=filtercat)
        
        print("Data filtered by " + str(filtercat))
       
    #If user input is "no" or "n" then stops user prompt.
    elif userinput.lower() == "no" or userinput.lower() == "n":
        print("OK no more filtering.")
        break
    
    #If user input is not "yes", "y", "no", or "n" then prompts for another input.
    else: 
        print("User Input invalid. Try Again.")
        
# Truncates data so that only the running times are visible
# Preparing data to be graphed
histo_data = numpy.array([(a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]) for (i,a) in numpy.ndenumerate(data)])


# In[5]:

### Step 3: Collect data from user and extrapolate if user data is incomplete

# 9 values for the x-axis, one for each km marker
race_bins = list(numpy.arange(5,45,5))
race_bins.append(42)

# array for prompting user input for each km marker
input_text = [str(i)+'k time: ' for i in race_bins]
input_text[len(input_text)-1] = 'Final Time: '

# Empty array to collect the user's running times via input
user_times = []

print ("Enter the time for each marker (as X:XX:XX). If no data, enter 'None'.")

# Loop to start collecting user inputted data
for i in input_text:
    user_text = input(i)
    
    # If user input is 'None', stop prompting for input
    if user_text.lower() == 'None'.lower():
        break
    else:
        # append to user_times array and also convert times to seconds
        user_times.append(timetosecs(user_text.encode('utf-8')))

# Extrapolate data using linear regression if data is incomplete
if len(user_times) != len(race_bins):
    
    # truncate user_x (km markers) for each user_time available
    user_x = race_bins[:len(user_times)]

    # Linear regression to determine slope and intercept for extrapolating user data
    slope, intercept, a, b, c = linregress(user_x, user_times)
    
    for i in range(len(user_times), len(race_bins)):
        user_times.append(int(slope*race_bins[i]+intercept))


# In[6]:

### Step 4: Preparing to graph - define refresh rate and number of iterations

# After one iteration, 500s will have passed in the marathon
refresh_rate = 500 
max_time = histo_data.max()

# Number of iterations is based on largest time.
iterations = int(plb.ceil(max_time/refresh_rate))


# In[7]:

### Step 5: Generate Cumulative Chart
# This is a chart that shows how many runners total have reached the km marker at each iteration.

# Clears figure
plb.clf()

# Start looping through each iteration
for i in range(int(iterations)):
    
    # Calculate the elapsed time in the marathon
    elapsed_time = refresh_rate * i + refresh_rate
    
    # Find the index of the user data array that correlates to the user's closest time (but under) the "elapsed time"
    user_bin = len([i for i in user_times if i < elapsed_time])-1
    
    # Element-by-element comparison of the running times and the elapsed time
    #Output is a boolean array that returns True/False if the runner's time is below the elapsed time
    comp_data = numpy.less_equal(histo_data, elapsed_time*numpy.ones(histo_data.shape))
    
    #sum of Trues along columns for bar graph counts.
    col_sum = numpy.sum(comp_data, axis=0)

    # Graphs the total number of people who reached the mile marker
    # Normal participants have cyan bars
    p = plb.bar(race_bins,col_sum, color='cyan', align='center', width=2, linewidth = 1)
    
    # User inputted bar is black
    p.patches[user_bin].set_color('k')
    
    # Set y-axis upper and lower limits
    plb.ylim(0, len(histo_data)*1.1)
    
    #Strings for x-axis, graph title and labels, legend, and grid.
    plb.title("Cumulative Chart")
    plb.xlabel("Kilometers")
    plb.ylabel("Number of People")
    
    # Customize legend
    cyan_patch = mpatches.Patch(color = 'cyan', label = 'Participants')
    black_patch = mpatches.Patch(color = 'black', label = 'User')
    plb.legend(handles=[black_patch, cyan_patch], loc= 'upper left', bbox_to_anchor=(1,1))
 
    # Pause and display chart on the same figure
    plb.grid(True)
    plb.pause(0.1)
    display.clear_output(wait=True)
    display.display(plb.gcf())

    # Clear figure to prepare for next iteration
    plb.clf()


# In[8]:

### Step 6: Generate Dynamic Chart
# This chart shows at each iteration how many people have reached that specific km marker.

# Clears figure
plb.clf()

# Start looping through each iteration
for i in range(int(iterations)):
    
    # Calculate the elapsed time in the marathon
    elapsed_time = refresh_rate * i + refresh_rate
    
    # Find the index of the user data array that correlates to the user's closest time (but under) the "elapsed time"
    user_bin = len([i for i in user_times if i < elapsed_time])-1
    
    # Same element-by-element comparison of the running times and the elapsed time
    #Output is a boolean array that returns True/False if the runner's time is below the elapsed time
    comp_data = numpy.less_equal(histo_data, elapsed_time*numpy.ones(histo_data.shape))
    
    # Sum of Trues along each row    
    row_sum = numpy.sum(comp_data, axis=1)
    
    # dyn_sum: outputs an array where each value in the row_sum list is counted up.
    # E.g. if row_sum[i] = 1 then that means the participant reached 5k only
    # dyn_sum collects a list (n=9) of the count of each of the values of the row_sums to determine which stage each participant is at
    dyn_sum = []
    for i in range(len(race_bins)):
        dyn_sum.append(list(row_sum).count(i+1))

    # Graphs the total number of people who reached the mile marker
    # Normal participants have cyan bars
    p = plb.bar(race_bins,dyn_sum, color='cyan', align='center', width=2, linewidth = 1)
    
    # User inputted bar is black
    p.patches[user_bin].set_color('k')
    
    # Set y-axis upper and lower limits
    plb.ylim(0, len(histo_data)*1.1)
    
    #Strings for x-axis, graph title and labels, legend, and grid.
    plb.title("Dynamic Chart")
    plb.xlabel("Kilometers")
    plb.ylabel("Number of People")
    
    # Customize legend
    cyan_patch = mpatches.Patch(color = 'cyan', label = 'Participants')
    black_patch = mpatches.Patch(color = 'black', label = 'User')
    plb.legend(handles=[black_patch, cyan_patch], loc= 'upper left', bbox_to_anchor=(1,1))
 
    # Pause and display chart on the same figure
    plb.grid(True)
    plb.pause(0.1)
    display.clear_output(wait=True)
    display.display(plb.gcf())

    # Clear figure to prepare for next iteration
    plb.clf()


# In[ ]:



