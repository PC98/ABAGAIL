#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#
export CLASSPATH=../ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot

# pima test
printf "Running pima test\n"
jython pima_test.py

# fourpeaks test
printf "\nRunning four peaks\n"
jython fourpeaks.py

# count ones test
printf "\nRunning count ones\n"
jython countones.py

# k-Coloring test
printf "\nRunning k-Coloring\n"
jython kcoloring.py

# curves
printf "\nPlotting curves\n"
python3 curve_plotter.py

printf "\nCurves created in ./data/plot\n"