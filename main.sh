#!/bin/bash

# may pivot to docker later
source ../shvenv/bin/activate

mkdir tmp

numfiles=${1:-5}
cd init
./init.sh $numfiles

cd ../core
./core.sh

cd ../eval
./eval.sh

deactivate