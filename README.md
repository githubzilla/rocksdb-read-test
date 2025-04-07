# RocksDB Read Performance Test

This project contains a benchmark test for evaluating read performance of RocksDB with different optimization techniques.

## Test Scenarios

The benchmark tests the following optimization techniques:

1. Baseline (No optimizations)
2. Multi-Get optimization
3. Block cache optimization
4. Prefix bloom filter optimization
5. Block cache size optimization (testing various cache sizes)
6. Combined optimizations (Block cache + Bloom filter + Multi-get)

## Building the Project

```bash
# Create build directory
mkdir -p build && cd build

# Generate build files
cmake ..

# Build the project
make

