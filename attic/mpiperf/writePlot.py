#!/usr/bin/env python
#
# ingestPlot.py
#
# TODO: Update the description below
#
# This is a simple plotter for the data coming out of tMSSink, the ingest MS writer simulation app
# that Max wrote to test the write IO issues we are having with scratch2.
#
# It reads in a file created by the tMSSink program, extracts the timing data out of it
# and plots the data.
#
# Y axis is write time in seconds
# X axis is the integration ordinal
#
# The data comes in the form:
#
# INFO  askap.tMSSink (0, galaxy-ingest07) [2016-08-29 16:42:15,453] - Received 763 integration(s) for rank=0
# INFO  askap.tMSSink (0, galaxy-ingest07) [2016-08-29 16:42:17,169] -    - mssink took 1.99 seconds
#
# Lines of the first type are discarded and the seconds value in the second line type are used as data for plot.
# regex is used to find and extract that value.
#
# So far the file processed is hard coded.
#
# Copyright: CSIRO 2017
# Author: Max Voronkov <maxim.voronkov@csiro.au>(original)
# Author: Eric Bastholm <eric.bastholm@csiro.au>
# Author: Paulus Lahur <paulus.lahur@csiro.au>
# Author: Stephen Ord <stephen.ord@csiro.au> (just added the table)

# import python modules
import argparse
import re
import numpy as np
import matplotlib
# Use non-interactive backend to prevent figure from popping up
# Note: Agg = Anti Grain Geometry
matplotlib.use('Agg')
import matplotlib.pyplot as pl


# find all lines that have "mssink took" followed by a group of the form n.nn
parser = argparse.ArgumentParser(
    description="Output the distribution of write times.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("logfile", help="mpiperf log file to use for analysis")
plotFileDefault = "plot.png"
parser.add_argument("-p", "--plotfile", help="Output plot file in png format", default=plotFileDefault)
args = parser.parse_args()
print("log file : " + args.logfile)
plotFile = args.plotfile

#reLine = re.compile("mssink took ([0-9]*(\.[0-9]*)?)")
reLine = re.compile("Wrote integration ([0-9]*) in ([0-9]*(\.[0-9]*)?) seconds")
# This is the data for /scratch2 run
with open(args.logfile) as fp:
    count = 0
    values = [];
    badCount = 0;
    min = 99
    max = 0
    for line in iter(fp.readline, ''):
        print(line)
        match = reLine.search(line)
        print(match)
        if match:
            count += 1
            secs = float("" + match.group(2))
            values.append(secs)
            print(secs)

            if secs < min:
               min = secs
            if secs > max:
               max = secs

N = len(values)
mean = sum(values)/N
bad = float(badCount)/N*100
sdev = np.std(values)
median = np.median(values)

print("statistics of integration time:")
print("integration time: min   : {0:.2f}".format(min))
print("integration time: max   : {0:.2f}".format(max))
print("integration time: mean  : {0:.2f}".format(mean))
print("integration time: median: {0:.2f}".format(median))
print("integration time: sdev  : {0:.2f}".format(sdev))
#print("integration time: bad   : {0:.2f}%".format(bad))

runningMean = np.convolve(values, np.ones((100,))/100, mode="valid")

# Plotting
timeMin = 0.0
timeMax = 1.5*max

pl.figure(1)

pl.subplot(211)
plot = pl.plot(values, "b.", label="times")
pl.axhline(sum(values)/N, color="red", label="average")
pl.xlim(0, N)
pl.ylim(timeMin, timeMax)
pl.title("Write time per integration")
pl.xlabel("integration cycle")
pl.ylabel("time (s)")
pl.legend(numpoints=1)

#pl.subplot(312)
#plot = pl.plot(runningMean, "r-", label="avg")
#pl.xlim(0, N)
#pl.ylim(0)
#pl.title("MSSink writing times average")
#pl.xlabel("integration")
#pl.ylabel("seconds")


pl.subplot(212)
#pl.hist(values, 40, alpha=0.5)
pl.hist(values, range=(timeMin, timeMax), bins=100)
pl.xlim(timeMin, timeMax)
pl.xticks(np.arange(timeMin, timeMax, 0.5))
pl.title("Distribution")
pl.xlabel("time (s)")
pl.ylabel("count")

pl.tight_layout()

# display the plot (must use interactive backend!)
#pl.show()

# save into file
pl.savefig(plotFile)
print("plot file: " + plotFile)
