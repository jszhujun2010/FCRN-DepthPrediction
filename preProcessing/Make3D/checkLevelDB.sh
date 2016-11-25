#!/bin/bash

data=$1
outdata=$2
label=$3
outlabel=$4
num=$5

echo $data
echo $outdata
echo $label
echo $outlabel

./checkLevelDB.py $data $outdata $num
./checkLevelDB.py $label $outlabel $num

sz $outdata
sz $outlabel
