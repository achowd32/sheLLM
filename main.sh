#!/bin/bash

#TODO: pivot to docker, ensure dependencies
source ../shvenv/bin/activate

#handle flags
while getopts "pn:" flag; do
 case $flag in
   p) # Handle the -p flag
    pretrained=1
    ;;
   n) # Handle the -n flag
    numfiles=${OPTARG}
    re='^[0-9]+$'
    if ! [[ $numfiles =~ $re ]]; then
        echo "Error: argument passed to -n is not a number"
        exit 1
    fi
    ;;
   \?)
    exit 1
    ;;
 esac
done

#create necessary directories
mkdir tmp

#initiate pipeline
numfiles=${numfiles:-5}
pretrained=${pretrained:-0}
cd init
./init.sh $numfiles $pretrained

cd ../core
./core.sh

cd ../eval
./eval.sh

deactivate