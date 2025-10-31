#!/bin/bash

clear

echo "+-----------------------------+"
echo "|  _______ ______  _____ _______ |"
echo "| |__   __|  ____|/ ____|__   __| |"
echo "|    | |  | |__  | (___    | |   |"
echo "|    | |  |  __|  \___ \   | |   |"
echo "|    | |  | |____ ____) |  | |   |"
echo "|    |_|  |______|_____/   |_|   |"
echo "+-----------------------------+"
echo "COO vs CSR SpMV Benchmark Runner"
echo ""

sleep 2

# default: no verbose
VERBOSE_FLAG=""

if [ "$1" = "--verbose" ]; then
    VERBOSE_FLAG="--verbose"
    echo "Running in verbose mode"
fi

echo ""

#valgrind --tool=cachegrind --cachegrind-out-file=../outputs/coo_cachegrind_output/cachegrind.out ./../outputs/spmv_coo ../data/cage14/cage14.mtx
./../outputs/spmv_coo $VERBOSE_FLAG ../data/cage14/cage14.mtx

echo ""

#valgrind --tool=cachegrind --cachegrind-out-file=../outputs/csr_cachegrind_output/cachegrind.out ./../outputs/spmv_csr ../data/cage14/cage14.mtx
./../outputs/spmv_csr $VERBOSE_FLAG ../data/cage14/cage14.mtx

echo ""

export OMP_NUM_THREADS=16
./../outputs/parallel_spmv_csr $VERBOSE_FLAG ../data/cage14/cage14.mtx

cd ../benchmarks
#python3 script.py --show-plot
python3 script.py