#!/bin/bash

#combines all csvs in a folder into one, excluding headers
for file in *.csv; do

    echo $file
    sed -n '2,8641p' $file >> subset.csv

done