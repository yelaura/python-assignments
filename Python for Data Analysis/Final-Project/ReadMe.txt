Platform: Anaconda3/Jupyter Notebook
System: Windows 7
Python Version: 3.6


Packages and dependencies
* Numpy
* Pylab
* Scipy (ceil, stats.linregress)
* IPython.display
* Matplotlib.patches


Sequence of how your code needs to be executed: From top to bottom
1. Import dependencies and define critical functions
2. Load Data from csv and remove rows with missing values
3. Filter Data (optional)
4. Collect data from user and extrapolate if user data is incomplete.
5. Preparing to graph - define refresh rate and number of iterations
6. Generate Cumulative Chart
7. Generate Dynamic Chart


Comments:
* CSV file “marathon_results_2017.csv” must be loaded from the same directory as the py file.
* Output of the figures in the last two sections may require code modifications since we used a Jupyter-specific module (ipython.display).