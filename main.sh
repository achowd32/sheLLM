#!/bin/bash

#TODO: pivot to docker, ensure dependencies
source ../shvenv/bin/activate

#handle flags
while getopts "pn:" flag; do
 case $flag in
   p) # Handle the -p flag (pretrained or not)
    pretrained=1
    ;;
   k) # Handle the -k flag (keep files or not)
    keep=1
    ;;
   n) # Handle the -n flag (number of files from dataset)
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
mkdir outputs

#initiate pipeline
numfiles=${numfiles:-5}
pretrained=${pretrained:-0}
cd init
./init.sh $numfiles $pretrained

cd ../core
./core.sh

cd ../eval
./eval.sh

#delete temporary files
cd ..
keep=${keep:-0}
if [[ "$keep" -eq 0 ]]; then
    rm -rf tmp/
fi

deactivate