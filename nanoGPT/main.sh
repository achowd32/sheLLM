#!/bin/bash

cd "$(dirname $0)"

mkdir data/ model/

cd init
./init.sh

cd ../core
./core.sh
