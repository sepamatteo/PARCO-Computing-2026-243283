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

#valgrind --tool=cachegrind --cachegrind-out-file=../outputs/coo_cachegrind_output/cachegrind.out ./../outputs/spmv_coo ../data/cage14/cage14.mtx
#./../outputs/spmv_coo --verbose ../data/cage14/cage14.mtx
./../outputs/spmv_coo ../data/cage14/cage14.mtx

#valgrind --tool=cachegrind --cachegrind-out-file=../outputs/csr_cachegrind_output/cachegrind.out ./../outputs/spmv_csr ../data/cage14/cage14.mtx
#./../outputs/spmv_csr --verbose ../data/cage14/cage14.mtx
./../outputs/spmv_csr ../data/cage14/cage14.mtx

export OMP_NUM_THREADS=16
#./../outputs/parallel_spmv_csr --verbose ../data/cage14/cage14.mtx
./../outputs/parallel_spmv_csr ../data/cage14/cage14.mtx

cd ../benchmarks
#python3 script.py --show-plot
python3 script.py