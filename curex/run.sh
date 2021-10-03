#!/bin/env bash
#set -xe

########## Args ##########
# 1: window size         #
# 2: slide size          #
# 3: k                   #
# 4: threshold           #
##########################

# find outliers using the GPU
./cu_od --input od_input.csv -c $(wc -l < od_input.csv) -d 1 -w $1 -s $2 -k $3 -t $4 #&>/dev/null

# find global outliers by sorting output and removing duplicates
sort -n outliers_out.txt | uniq > outliers.csv

# replace python bin if necessary (pandas lib is required)
/opt/anaconda/bin/python query.py

# delete intermediate files
rm outliers_out.txt outliers.csv

