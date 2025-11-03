#!/bin/bash

# default: no verbose no plot
VERBOSE_FLAG=""
PLOT_FLAG=""
CACHEGRIND_FLAG=""
MATRIX_FILE="../data/cage14/cage14.mtx"     # default matrix file
RUN_BENCHMARK=""

RUN_COO=""      #--coo
RUN_SEQ_CSR=""  #--seq-csr
RUN_PAR_CSR=""  #--par-sqr

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose)      VERBOSE_FLAG="--verbose"; shift ;;
        --show-plot)    PLOT_FLAG="--show-plot"; shift ;;
        --cachegrind)   CACHEGRIND_FLAG="1"; shift ;;
        --matrix)
            [[ -z "${2:-}" ]] && { echo "Error: --matrix needs a path" >&2; exit 1; }
            MATRIX_FILE="$2"; shift 2
            ;;
        --benchmark)    RUN_BENCHMARK="1"; shift ;;
        --coo)          RUN_COO="1"; shift ;;
        --seq-csr)      RUN_SEQ_CSR="1"; shift ;;
        --par-csr)      RUN_PAR_CSR="1"; shift ;;
        -*)
            echo "Warning: unknown option $1" >&2
            shift
            ;;
        *) shift ;;   # ignore positional args
    esac
done

if [[ -z "$RUN_COO$RUN_SEQ_CSR$RUN_PAR_CSR" ]]; then
    # none specified → run everything
    RUN_COO="1"; RUN_SEQ_CSR="1"; RUN_PAR_CSR="1"
fi

if [[ ! -f "$MATRIX_FILE" ]]
then
    echo "Error: Matrix file not found: $MATRIX_FILE" >&2
    exit 1
fi

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

run_cachegrind() {
    local out_dir="$1"      # where cachegrind.out will be written
    local cmd=( "${@:2}" )  # the real command + its args

    if [[ -n "$CACHEGRIND_FLAG" ]]; then
        mkdir -p "$out_dir"
        valgrind --tool=cachegrind --cache-sim=yes\
                 --cachegrind-out-file="$out_dir/cachegrind.out" \
                 "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

echo "Using matrix: $MATRIX_FILE"
echo ""

echo ""

if [[ -n "$RUN_COO" ]]; then
    run_cachegrind \
        ../outputs/coo_cachegrind_output \
        ./../outputs/spmv_coo $VERBOSE_FLAG $MATRIX_FILE
        echo ""
fi

if [[ -n "$RUN_SEQ_CSR" ]]; then
    run_cachegrind \
        ../outputs/csr_cachegrind_output \
        ./../outputs/spmv_csr $VERBOSE_FLAG $MATRIX_FILE
        echo ""
fi

export OMP_NUM_THREADS=16
if [[ -n "$RUN_PAR_CSR" ]]; then
    run_cachegrind \
        ../outputs/par_csr_cachegrind_output \
        ./../outputs/parallel_spmv_csr $VERBOSE_FLAG $MATRIX_FILE
        echo ""
fi

cd ../benchmarks
python3 script.py $PLOT_FLAG