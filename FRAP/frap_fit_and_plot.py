import numpy, scipy, matplotlib
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import csv
import pandas as pd
import sys
import math
import os
from math import log10, floor

##############
# IN THIS VERSION OF THE SCRIPT, THE MINIMA IS ANCHORED TO THE DATA MINIMA
##############

# do not print unnecessary warnings during curve_fit()
warnings.filterwarnings("ignore")

file = ("data.txt")

#####################
#####USER INPUTS#####
#####################
# ask user whether they want to fit to the whole dataset
# if y, define the number of seconds post-bleach you want to fit the curve to
part_fit = input('\nDo you want to fit to a part of the data? (y/n): ')
if part_fit == 'y':
    part_fit_x = float(input('How many seconds from t=0 would you like to fit to?: '))
# ask the user to input predicted half time, this is used in the initial pararemeters for the curve fit
predictedhalf = float(input('What is the predicted half life? (s): '))
# ask the user to input number of replicates
numsamples = float(input('How many replicates does the average represent?: '))
# ask user for colour of plot, combined with the tab colour pallete for matplotlib
colour = input('What colour would you like to plot the data points as?: ')
# ask user what binning factor they want to use
# the binning factor is only used in the plotting process, to 'declutter' the plot, all points are used for the fitting
bin_no = float(input('What binning factor would you like to use for the plot?: '))
# ask the user if both the SEM and SD are provided
two_errors = input('Are both the SEM and SD provided? (y/n): ')

############################
# READING AND FILTERING DATA#
############################
# read in the dataframe, with time in col1, intensity in col2, SD/SEM in col3
df = pd.read_csv(file, sep='\t', header=None)
# extract data for pre and post-bleach events
# take column 1 (time) and filter to events after (and inlcuding) t=0 for post-bleach
# take column 1 and filter to pre-bleach (t<0)
postbleachdf = df[df.iloc[:, 0] >= 0]
prebleachdf = df[df.iloc[:, 0] < 0]
# from the post-bleach dataset, define time, intensity and error
time = postbleachdf.iloc[:, 0]
intensity = postbleachdf.iloc[:, 1]
error = postbleachdf.iloc[:, 2]
# extract data from pre-bleach csv
time2 = prebleachdf.iloc[:, 0]
intensity2 = prebleachdf.iloc[:, 1]
error2 = prebleachdf.iloc[:, 2]
# optional for fitting to first part of curve
# if user input defines, filter the post bleach dataset to that many frames
if part_fit == 'y':
    # calculate the number of frames to fit to using the timestep from the first 2 data points in the time column
    part_fit_no = part_fit_x / (postbleachdf.iloc[1, 0] - postbleachdf.iloc[0, 0])
    # define the time and intensity datapoints from the above calculated number of frames
    time_part = postbleachdf.iloc[0:int(part_fit_no), 0]
    intensity_part = postbleachdf.iloc[0:int(part_fit_no), 1]
# binning (optional)
# if there are too many data points to plot, some can be binned for plotting
# takes the user-inputted value as the binning factor
# this is only for plotting and not for the model fit
time_binned = time[::int(bin_no)]
intensity_binned = intensity[::int(bin_no)]
error_binned = error[::int(bin_no)]

prebleach_last = prebleachdf.tail(1)
postbleach_first = postbleachdf.head(1)

bridge = pd.concat([prebleach_last, postbleach_first], axis=0)
bridge_time = bridge.iloc[:, 0]
bridge_intensity = bridge.iloc[:, 1]

# if the user has supplied both the SEM and SD, they are both plotted on the graph
# SEM is the third column, SD is designated as the fourth column
# the only consequence of this is the darkness of the error area on the plot
if two_errors == 'y':
    # postbleach second error
    error_2 = postbleachdf.iloc[:, 3]
    error_2_binned = error_2[::int(bin_no)]
    # prebleach second error
    error2_2 = prebleachdf.iloc[:, 3]


########################
##FUNCTIONS USED BELOW##
########################
# define the one-phase association equation
# min(intensity) is the frap minimum (usually ~0.2-0.5), taken from the post-bleach data
# b is the plateau (1 for fully recovering droplet)
# K is the rate constant (t1/2 = ln(2)/K)
# x is the time point
def func(x, b, K):
    return ((min(intensity)) + ((b - (min(intensity))) * (1 - numpy.exp(-K * x))))


# simple linear model for fitting to the pre-bleach data
def linear_func(x, m, c):
    return (m * x + c)


# define rounding function, used in outputs
def round_it(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


# define the initial parameters for the model fit
# b is the frap maximum from the data
# K is the calculated rate constant from the user-inputted thalf)
initialParameters = numpy.array([max(intensity), (math.log(2) / predictedhalf)])

#####################
####MODEL FITTING####
#####################
# complete the curve fitting over the whole  unbinned dataset
fittedParameters, pcov = curve_fit(func, time, intensity, initialParameters)
modelPredictions = func(time_binned, *fittedParameters)
absError = modelPredictions - intensity
SE = numpy.square(absError)  # squared errors
MSE = numpy.mean(SE)  # mean squared errors
RMSE = numpy.sqrt(MSE)  # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (numpy.var(absError) / numpy.var(intensity))
time_str = round_it(float(max(time)), 4)
# if specified by the user, complete curve fitting to the initial part of the curve also
if part_fit == 'y':
    ## fo the initial fitting
    fittedParameters_initial, pcov = curve_fit(func, time_part, intensity_part, initialParameters)
    modelPredictions_initial = func(time_part, *fittedParameters_initial)
    absError_initial = modelPredictions_initial - intensity_part
    SE_initial = numpy.square(absError_initial)  # squared errors
    MSE_initial = numpy.mean(SE_initial)  # mean squared errors
    RMSE_initial = numpy.sqrt(MSE_initial)  # Root Mean Squared Error, RMSE
    Rsquared_initial = 1.0 - (numpy.var(absError_initial) / numpy.var(intensity_part))
    time_part_str = round_it(float(max(time_part)), 4)

# curve fit the pre-bleach data
init_params = numpy.array([-1, 1])
fit_params, pcov = curve_fit(linear_func, time2, intensity2, init_params)
model_prediction = func(time2, *fit_params)

print('\nPlotting data for ' + str(file))


#####################
##PLOTTING FUNCTION##
#####################
def ModelAndScatterPlot(graphWidth, graphHeight):
    # set global plot font for the figure
    plt.rcParams['font.size'] = '16'

    # define the figure that the plots will be written to
    f = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=300)
    axes = f.add_subplot(111)

    # plt.xscale("log", base=2)
    # plt.yscale('log', base=10)

    # set the axes limits based on the bounds of the unbinned data
    axes.set_xlim([(min(time2) - 10), (max(time) + 10)])
    axes.set_ylim([(min(intensity) - 0.05), (max(intensity2) + (max(intensity2) * 0.075))])

    # plot the binned data as a scatter
    # plot the pre-bleach data also
    # take the colour from the user input and combine with the tab palette
    axes.plot(time_binned, intensity_binned, color='tab:' + str(colour), markersize='10', alpha=0.75,
              markeredgewidth=0.0)
    axes.plot(time2, intensity2, color='tab:' + str(colour), markersize='10', alpha=0.75, markeredgewidth=0.0)

    # if a second error column is specified, plot that first
    if two_errors == 'y':
        plt.fill_between(time_binned, intensity_binned - error_2_binned, intensity_binned + error_2_binned,
                         color='tab:' + str(colour), alpha=0.2)
        plt.fill_between(time2, intensity2 - error2_2, intensity2 + error2_2, color='tab:' + str(colour), alpha=0.2)
    # plot the first error colum (SEM)
    # plot the post-bleach
    plt.fill_between(time_binned, intensity_binned - error_binned, intensity_binned + error_binned,
                     color='tab:' + str(colour), alpha=0.4)
    # plot the pre-bleach
    plt.fill_between(time2, intensity2 - error2, intensity2 + error2, color='tab:' + str(colour), alpha=0.4)

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(time), max(time))
    yModel = func(xModel, *fittedParameters)
    # plot the line model
    axes.plot(xModel, yModel, color='black', label='Complete (c): (0-' + str(time_str) + ' s)')
    # if a second fit is specified, create the data and plot that also
    if part_fit == 'y':
        xModel_initial = numpy.linspace(min(time_part), max(time_part))
        yModel_initial = func(xModel_initial, *fittedParameters_initial)
        axes.plot(xModel_initial, yModel_initial, color='black', linestyle='dashed',
                  label='Partial (p): (0-' + str(time_part_str) + ' s)')

    # plot the two
    axes.plot(bridge_time, bridge_intensity, color='tab:' + str(colour), markeredgewidth=0.0, alpha=0.75)

    # create data and plot linear fit of pre-bleach data
    xMod = numpy.linspace(min(time2), max(time2))
    yMod = linear_func(xMod, *fit_params)
    axes.plot(xMod, yMod, color='tab:' + str(colour), markeredgewidth=0.0, alpha=0.75)

    # set the axes labels
    axes.set_xlabel('Time after bleach (s)')
    axes.set_ylabel('Normalised Intensity')

    # calculate the stats for the output on the graph
    thalf = math.log(2) / fittedParameters[1]
    thalfround = (round_it(float(thalf), 3))
    thalfoutstr = "T₀.₅ = " + str(thalfround) + " s"
    r2 = str(Rsquared)
    r2round = (round_it(float(r2), 2))
    r2outstr = "R² = " + str(r2round)
    plateauround = round_it(fittedParameters[0], 3)
    plateauoutstr = "Plateau = " + str(plateauround)
    minimum = min(intensity)
    minround = round_it(float(minimum), 2)
    minoutstr = "Minimum = " + str(minround)
    numoutstr = "n = " + str(numsamples)
    span = fittedParameters[0] - minimum
    # if a second fit was completed, also calculate the stats for those
    if part_fit == 'y':
        r2_initial = str(Rsquared_initial)
        r2round_initial = (round_it(float(r2_initial), 2))
        r2outstr_initial = "R² = " + str(r2round_initial)
        half_initial = minimum + (span / 2)
        thalf_initial = (math.log(1 - ((half_initial - minimum) / ((fittedParameters_initial[0]) - minimum))) / -
        fittedParameters_initial[1])
        thalfround_initial = (round_it(float(thalf_initial), 3))
    # draw the t0.5 calculation lines from the model
    axes.vlines([thalf], 0, [minimum + (span / 2)], linestyles='solid', colors='gray')
    axes.hlines([minimum + (span / 2)], (min(time2) - 10), thalf, linestyles='solid', colors='gray')
    if part_fit == 'y':
        axes.vlines([thalf_initial], 0, [half_initial], linestyles='dashed', colors='gray')
        axes.hlines([half_initial], (min(time2) - 10), thalf_initial, linestyles='dashed', colors='gray')

    # add text to the graph
    if part_fit == 'y':
        plt.text(0.12, 0.95, "T₀.₅ = " + str(thalfround_initial) + " s (p), " + str(thalfround) + " s (c)", fontsize=16,
                 transform=axes.transAxes, weight='bold')
        plt.text(0.12, 0.80, "R² = " + str(r2round_initial) + " (p), " + str(r2round) + " (c)", fontsize=16,
                 transform=axes.transAxes)
        plt.text(0.12, 0.90, plateauoutstr, fontsize=16, transform=axes.transAxes)
        plt.text(0.12, 0.85, minoutstr, fontsize=16, transform=axes.transAxes)
        plt.text(0.12, 0.75, numoutstr, fontsize=16, transform=axes.transAxes)
        plt.legend(loc='lower right')
    else:
        plt.text(0.12, 0.95, thalfoutstr, fontsize=16, transform=axes.transAxes, weight='bold')
        plt.text(0.12, 0.80, r2outstr, fontsize=16, transform=axes.transAxes)
        plt.text(0.12, 0.90, plateauoutstr, fontsize=16, transform=axes.transAxes)
        plt.text(0.12, 0.85, minoutstr, fontsize=16, transform=axes.transAxes)
        plt.text(0.12, 0.75, numoutstr, fontsize=16, transform=axes.transAxes)
        plt.legend(loc='lower right')

    # add axes lines
    axes.spines['top'].set_visible(True)
    axes.spines['right'].set_visible(True)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)

    # if multiple fits have been completed, modify the legend to add a title
    axes.legend(title='Exponential fits:', loc='lower right')

    # save the figure as a pdf
    plt.savefig(file + '_plot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')  # clean up after using pyplot

    # print the spiel in the console
    print('')
    print('Fitted parameters for 0-' + str(round_it(float(max(time)), 3)) + ' s: ')
    print('Minimum value: ', minround)
    print('Plateau value: ', plateauround)
    print('t1/2: ', thalfround)
    print('R-squared: ', r2round)
    if part_fit == 'y':
        print('')
        print('Fitted parameters for 0-' + str(round_it(float(max(time_part)), 3)) + ' s: ')
        print('t1/2: ', thalfround_initial)
        print('R-squared: ', r2round_initial)


graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)

print('\nSaved plot successfully!\n')