#!/bin/bash

#navigate to appropriate directory
cd "$(dirname $0)"

#TODO: pivot to docker, ensure dependencies
source ../shvenv/bin/activate

#handle flags
while getopts "pdn:" flag; do
 case $flag in
   p) # Handle the -p flag (pretrained or not)
    pretrained=1
    ;;
   d) # Handle the -d flag (delete files or not)
    delete=1
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

#erase previous model and outputs
rm -rf model/ outputs/

#create necessary directories
[ -d "data" ] || mkdir data
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
delete=${delete:-0}
if [[ "$delete" -eq 1 ]]; then
    rm -rf data/ outputs/
fi

deactivate