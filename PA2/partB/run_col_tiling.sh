#!/bin/bash

# ========= CONFIG =========
SOURCE="tiling_col.cu"
EXEC="tiling_col"
SIZES=(1024 2048 4096 8192 16384 32768)
# SIZES=(1024 )
# ==========================

echo "Compiling CUDA program..."

nvcc -O3 $SOURCE -o $EXEC

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

echo "Compilation successful."
echo "----------------------------------"

for N in "${SIZES[@]}"
do
    echo "Running profiler for N = $N"

    # nsys nvprof ./$EXEC $N
#    nsys nvprof --print-gpu-trace ./$EXEC $N
./$EXEC $N

    echo "Finished N = $N"
    echo "----------------------------------"
done

echo "All runs completed."

rm -rf report*