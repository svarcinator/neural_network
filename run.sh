#!/bin/bash
## change this file to your needs

echo "Adding some modules"

#module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -Wall -std=c++17 -fopenmp -O3 -march=native -ffast-math src/*.cpp   -o network


echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./network
