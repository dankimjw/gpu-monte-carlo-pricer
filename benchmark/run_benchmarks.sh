#!/bin/bash
# Automated benchmark sweep for MC Option Pricing Engine
# Outputs CSV files for analysis

BIN=./mc_pricer
OUT_DIR=output

mkdir -p $OUT_DIR

echo "================================================================"
echo "  Block-Size Sweep (1M paths, 252 steps)"
echo "================================================================"

CSV=$OUT_DIR/benchmark_blocksize.csv
echo "block_size,kernel_ms,total_ms,price,error_pct,throughput_mpps" > $CSV

for BS in 32 64 128 256 512 1024; do
    echo -n "  Block size $BS ... "
    LINE=$($BIN --no-cpu -b $BS -n 1000000 2>&1)
    KERNEL=$(echo "$LINE" | grep "Kernel:" | awk '{print $2}')
    TOTAL=$(echo "$LINE" | grep "Total:" | awk '{print $2}')
    PRICE=$(echo "$LINE" | grep "GPU Monte Carlo" | tail -1 | awk '{print $2}')
    ERROR=$(echo "$LINE" | grep "Error vs BS:" | tail -1 | awk '{print $4}' | tr -d '%')
    THRU=$(echo "$LINE" | grep "Throughput:" | awk '{print $2}')
    echo "$BS,$KERNEL,$TOTAL,$PRICE,$ERROR,$THRU" >> $CSV
    echo "kernel=${KERNEL}ms  total=${TOTAL}ms  throughput=${THRU} Mpps"
done

echo ""
echo "Saved: $CSV"
echo ""

echo "================================================================"
echo "  Path-Count Sweep (block=256, 252 steps)"
echo "================================================================"

CSV=$OUT_DIR/benchmark_paths.csv
echo "num_paths,kernel_ms,total_ms,price,error_pct,std_err" > $CSV

for N in 1000 10000 100000 1000000 5000000 10000000; do
    echo -n "  Paths $N ... "
    LINE=$($BIN --no-cpu -b 256 -n $N 2>&1)
    KERNEL=$(echo "$LINE" | grep "Kernel:" | awk '{print $2}')
    TOTAL=$(echo "$LINE" | grep "Total:" | awk '{print $2}')
    PRICE=$(echo "$LINE" | grep "Price:" | tail -1 | awk '{print $2}' | tr -d '$')
    ERROR=$(echo "$LINE" | grep "Error vs BS:" | tail -1 | awk '{print $4}' | tr -d '%')
    STDERR=$(echo "$LINE" | grep "Std Err:" | awk '{print $3}' | tr -d '$')
    echo "$N,$KERNEL,$TOTAL,$PRICE,$ERROR,$STDERR" >> $CSV
    echo "kernel=${KERNEL}ms  price=\$${PRICE}  error=${ERROR}%"
done

echo ""
echo "Saved: $CSV"
echo ""

echo "================================================================"
echo "  CPU vs GPU Comparison"
echo "================================================================"

CSV=$OUT_DIR/benchmark_cpu_gpu.csv
echo "num_paths,cpu_ms,gpu_total_ms,speedup" > $CSV

for N in 10000 100000 1000000; do
    echo -n "  Paths $N ... "
    LINE=$($BIN -b 256 -n $N 2>&1)
    CPU_MS=$(echo "$LINE" | grep "Time:" | head -1 | awk '{print $2}')
    GPU_MS=$(echo "$LINE" | grep "Total:" | awk '{print $2}')
    SPEEDUP=$(echo "$LINE" | grep "Speedup:" | awk '{print $2}' | tr -d 'x')
    echo "$N,$CPU_MS,$GPU_MS,$SPEEDUP" >> $CSV
    echo "cpu=${CPU_MS}ms  gpu=${GPU_MS}ms  speedup=${SPEEDUP}x"
done

echo ""
echo "Saved: $CSV"
echo ""
echo "All benchmarks complete! CSVs in $OUT_DIR/"
