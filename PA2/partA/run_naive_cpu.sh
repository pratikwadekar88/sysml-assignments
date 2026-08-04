#!/bin/bash

SIZES=(1024 2048 4096 8192 16384)

echo "Compiling CPU program..."
g++ -O3 matmal_cpu.cpp -o matmal_cpu

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful."
echo "=============================="

for N in "${SIZES[@]}"
do
    echo "Running N = $N"
    ./matmal_cpu $N
    echo "=============================="
done