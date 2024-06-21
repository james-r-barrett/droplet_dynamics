import sys
import pandas as pd
import numpy
import math
from scipy.optimize import curve_fit
import statistics
import warnings

# do not print unnecessary warnings during curve_fit()
warnings.filterwarnings("ignore")

file = "data_individual.txt"

# read in dataframe
df=pd.read_csv(file, sep ='\t', header = None)

# ask the user to input predicted half time
predictedhalf = float(input('What is the estimated Thalf?: '))
full_scale = input('Is the data full-scale normalised? (y/n): ')

# extract data for pre and post-bleach events
# take column 1 (time) and filter to events after t=0 for post-bleach
# take column 1 and filter to pre-bleach (t<0)
postbleachdf = df[df.iloc[:,0]>=0]

# from the post-bleach dataset, define time, intensity and error
time = postbleachdf.iloc[:,0]

def func(x, a, b, K):
    return  a + (b-a) * (1 - numpy.exp(-K*x))

def func2(x, a, b, K):
    return (b) * (1 - numpy.exp(-K*x))

thalf_vales = []

for i in range(1,len(df.columns)):
    intensity = postbleachdf.iloc[:,i]

    # define the initial parameters for the model fit
    # a is the frap minmum from the data
    # b is the frap maximum from the data
    # K is the calculated rate constant from the user-inputted thalf)
    initialParameters = numpy.array([(min(intensity)), (max(intensity)), (math.log(2)/predictedhalf)])

    if full_scale == "y":
        fittedParameters, pcov = curve_fit(func2, time, intensity, initialParameters)
    else:
        fittedParameters, pcov = curve_fit(func, time, intensity, initialParameters)

    # run the function against each time point using the fitted parameters
    #modelPredictions = func(time_binned, *fittedParameters)


    K=str(fittedParameters[2])
    print(K)
    thalf = math.log(2)/float(K)

    thalf_vales.append(thalf)

    print("The fitted Thalf value for replicate "+str(i)+ " is: "+str(round(thalf,1))+ " seconds.")
    print("The fitted frap minimum for replicate "+str(i)+ " was: " +str(round(fittedParameters[0],3)))

thalf_mean = statistics.mean(thalf_vales)
thalf_sd = statistics.stdev(thalf_vales)
thalf_sem = thalf_sd/(math.sqrt((len(df.columns)-1)))

print("The mean Thalf value is: "+str(round(thalf_mean,1)))
print("The S.D. value is: "+str(round(thalf_sd,3)))
print("The S.E.M value is: "+str(round(thalf_sem,3)))