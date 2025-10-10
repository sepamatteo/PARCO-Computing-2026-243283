#!/bin/bash

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

./../outputs/spmv_coo ../data/cage14/cage14.mtx 
sleep 1 
./../outputs/spmv_csr ../data/cage14/cage14.mtx

cd ../benchmarks
python3 script.py