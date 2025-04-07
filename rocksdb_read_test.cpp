/**
 * @file RocksDB read test
 * @brief This file contains a test for reading data from a RocksDB database.
 *        It will try different optimizations and configurations to boost the
 * read performance.
 */

/**
 * @brief Test schema
 * 1. Create a RocksDB database.
 * 2. Insert a large number of key-value pairs into the database.
 * 3. Close the database.
 * 4. Reopen the database.
 * 5. Perform multiple threads random read operation on the database without any
 * optimizations.
 * 6. Perform multiple threads random read operation with multi-get
 * optimization.
 * 7. Perform multiple threads random read operation with block cache
 * optimization.
 * 8. Perform multiple threads random read operation with prefix bloom filter
 * optimization.
 * 9. Perform multiple threads random read operation with block cache size
 * optimization.
 */

#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/statistics.h"
#include "rocksdb/table.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Configuration parameters
const std::string DB_PATH = "./rocksdb_test_db";
const int NUM_KEYS = 100000000; // 10 million keys
const int VALUE_SIZE = 128;     // 128 values
const int NUM_THREADS = 8;      // Number of threads for concurrent reads
const int NUM_OPERATIONS_PER_THREAD = 10000; // Operations per thread
const int PREFIX_LENGTH = 8; // Length of prefix for bloom filter optimization
const int NUM_MULTI_GET =
    100; // Number of keys to fetch in one multi-get operation

// Block cache sizes to test
const std::vector<size_t> BLOCK_CACHE_SIZES = {
    64 * 1024 * 1024,  // 64 MB
    256 * 1024 * 1024, // 256 MB
    1024 * 1024 * 1024 // 1 GB
};

// Random generator for key selection
std::mt19937 rng(std::random_device{}());

/**
 * @brief Generate a random string of specified length
 */
std::string generate_random_string(size_t length) {
  const char charset[] =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::string result;
  result.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    result += charset[rng() % (sizeof(charset) - 1)];
  }

  return result;
}

/**
 * @brief Generate a key with format: prefix + sequential number
 */
std::string generate_key(int index, int prefix_length = PREFIX_LENGTH) {
  std::string prefix = generate_random_string(prefix_length);
  return prefix + "_" + std::to_string(index);
}

/**
 * @brief Create a database and populate it with test data
 */
std::vector<std::string> setup_database() {
  // Clean up any existing database
  rocksdb::DestroyDB(DB_PATH, rocksdb::Options());

  // Create database with default options
  rocksdb::Options options;
  options.create_if_missing = true;
  options.IncreaseParallelism(std::thread::hardware_concurrency());
  options.OptimizeLevelStyleCompaction();

  // Disable automatic compaction during initial data loading
  options.disable_auto_compactions = true;
  options.target_file_size_base = 128 * 1024 * 1024; // 64 MB
  options.level0_slowdown_writes_trigger = 50;
  options.level0_stop_writes_trigger = 64;

  rocksdb::DB *db_raw;
  rocksdb::Status status = rocksdb::DB::Open(options, DB_PATH, &db_raw);
  std::unique_ptr<rocksdb::DB> db(db_raw);

  if (!status.ok()) {
    std::cerr << "Unable to open database: " << status.ToString() << std::endl;
    exit(1);
  }

  std::cout << "Generating " << NUM_KEYS << " key-value pairs..." << std::endl;

  // Generate random keys and values
  std::vector<std::string> keys;
  keys.reserve(NUM_KEYS);

  for (int i = 0; i < NUM_KEYS; ++i) {
    keys.push_back(generate_key(i));

    if (i % 100000 == 0 && i > 0) {
      std::cout << "Generated " << i << " keys..." << std::endl;
    }
  }

  // Insert the key-value pairs using batch writes
  rocksdb::WriteOptions write_options;
  write_options.disableWAL = true; // Disable Write-Ahead Logging for speed
  std::string value = std::string(VALUE_SIZE, 'X');

  std::cout << "Inserting data into the database using batch writes..."
            << std::endl;

  // Configure batch size - adjust based on your system memory
  const int BATCH_SIZE = 10000;

  for (int i = 0; i < NUM_KEYS; i += BATCH_SIZE) {
    rocksdb::WriteBatch batch;
    int end = std::min(i + BATCH_SIZE, NUM_KEYS);

    for (int j = i; j < end; ++j) {
      batch.Put(keys[j], value);
    }

    status = db->Write(write_options, &batch);

    if (!status.ok()) {
      std::cerr << "Failed to write batch starting at index " << i
                << ", error: " << status.ToString() << std::endl;
      exit(1);
    }

    if (i % 100000 == 0 && i > 0) {
      std::cout << "Inserted " << i << " key-value pairs..." << std::endl;
    }
  }

  std::cout << "Database setup completed successfully with " << NUM_KEYS
            << " keys." << std::endl;

  // Perform manual compaction to optimize read performance
  std::cout << "Performing manual compaction..." << std::endl;
  rocksdb::CompactRangeOptions compact_options;
  compact_options.bottommost_level_compaction =
      rocksdb::BottommostLevelCompaction::kForce;
  status = db->CompactRange(compact_options, nullptr, nullptr);

  if (!status.ok()) {
    std::cerr << "Manual compaction failed: " << status.ToString() << std::endl;
  } else {
    std::cout << "Manual compaction completed successfully." << std::endl;
  }

  // Shuffle the keys to ensure they are not in order
  std::shuffle(keys.begin(), keys.end(), rng);

  return keys; // Return the list of keys for read operations
  // Database will be closed when db goes out of scope
}

/**
 * @brief Perform random read operations
 *
 * @param db Pointer to the database
 * @param keys List of keys to read from
 * @param num_ops Number of operations to perform
 * @return Duration in microseconds
 */
uint64_t perform_random_reads(rocksdb::DB *db,
                              const std::vector<std::string> &keys,
                              int num_ops) {
  std::uniform_int_distribution<int> dist(0, keys.size() - 1);
  rocksdb::ReadOptions read_options;
  std::string value;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_ops; ++i) {
    int key_idx = dist(rng);
    rocksdb::Status status = db->Get(read_options, keys[key_idx], &value);

    if (!status.ok() && !status.IsNotFound()) {
      std::cerr << "Error reading key: " << keys[key_idx]
                << ", status: " << status.ToString() << std::endl;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                               start_time)
      .count();
}

/**
 * @brief Perform random multi-get operations
 *
 * @param db Pointer to the database
 * @param keys List of keys to read from
 * @param num_ops Number of operations to perform (batches)
 * @return Duration in microseconds
 */
uint64_t perform_multiget_reads(rocksdb::DB *db,
                                const std::vector<std::string> &keys,
                                int num_ops) {
  std::uniform_int_distribution<int> dist(0, keys.size() - NUM_MULTI_GET);
  rocksdb::ReadOptions read_options;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_ops; ++i) {
    int start_idx = dist(rng);
    std::vector<rocksdb::Slice> key_slices;
    std::vector<std::string> values;
    std::vector<rocksdb::Status> statuses;

    key_slices.reserve(NUM_MULTI_GET);

    for (int j = 0; j < NUM_MULTI_GET; ++j) {
      key_slices.push_back(keys[start_idx + j]);
    }

    statuses = db->MultiGet(read_options, key_slices, &values);

    for (size_t j = 0; j < statuses.size(); ++j) {
      if (!statuses[j].ok() && !statuses[j].IsNotFound()) {
        std::cerr << "Error in multi-get for key: " << key_slices[j].ToString()
                  << ", status: " << statuses[j].ToString() << std::endl;
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                               start_time)
      .count();
}

/**
 * @brief Worker function for multi-threaded reads
 *
 * @param db Pointer to the database
 * @param keys List of keys to read from
 * @param num_ops Number of operations per thread
 * @param use_multiget Whether to use multi-get or single get
 * @param duration Atomic variable to accumulate duration
 */
void worker_thread(rocksdb::DB *db, const std::vector<std::string> &keys,
                   int num_ops, bool use_multiget,
                   std::atomic<uint64_t> &duration) {

  uint64_t thread_duration;

  if (use_multiget) {
    thread_duration = perform_multiget_reads(db, keys, num_ops / NUM_MULTI_GET);
  } else {
    thread_duration = perform_random_reads(db, keys, num_ops);
  }

  duration.fetch_add(thread_duration, std::memory_order_relaxed);
}

/**
 * @brief Perform multi-threaded reads with the given options
 *
 * @param options Database options to use when opening the DB
 * @param use_multiget Whether to use multi-get optimization
 * @param test_name Name of the test for reporting
 * @return Average operations per second
 */
double run_benchmark(const rocksdb::Options &options, bool use_multiget,
                     const std::string &test_name,
                     std::vector<std::string> &keys) {
  // Open database
  rocksdb::DB *db_raw;
  rocksdb::Status status = rocksdb::DB::Open(options, DB_PATH, &db_raw);
  std::unique_ptr<rocksdb::DB> db(db_raw);

  if (!status.ok()) {
    std::cerr << "Unable to open database: " << status.ToString() << std::endl;
    exit(1);
  }

  // Launch worker threads
  std::vector<std::thread> threads;
  std::atomic<uint64_t> total_duration(0);

  std::cout << "Running benchmark: " << test_name << "..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back(worker_thread, db_raw, keys, NUM_OPERATIONS_PER_THREAD,
                         use_multiget, std::ref(total_duration));
  }

  // Wait for threads to finish
  for (auto &thread : threads) {
    thread.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  uint64_t wall_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              end_time - start_time)
                              .count();

  // Calculate statistics
  double avg_thread_time_us = total_duration.load() / NUM_THREADS;
  double avg_latency_us = avg_thread_time_us / NUM_OPERATIONS_PER_THREAD;
  double total_ops = NUM_THREADS * NUM_OPERATIONS_PER_THREAD;
  double ops_per_sec = total_ops / (wall_time_us / 1000000.0);

  // Print results
  std::cout << "----------------------------------------" << std::endl;
  std::cout << test_name << " Results:" << std::endl;
  std::cout << "  Total time: " << wall_time_us / 1000000.0 << " seconds"
            << std::endl;
  std::cout << "  Average thread time: " << avg_thread_time_us / 1000.0 << " ms"
            << std::endl;
  std::cout << "  Average operation latency: " << avg_latency_us << " us"
            << std::endl;
  std::cout << "  Operations per second: " << std::fixed << std::setprecision(2)
            << ops_per_sec << std::endl;

  // Print cache statistics if available
  if (options.statistics) {
    std::cout
        << "  Block cache hit ratio: "
        << options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_HIT) *
               100.0 /
               (options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_HIT) +
                options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_MISS))
        << "%" << std::endl;
  }

  std::cout << "----------------------------------------" << std::endl;

  return ops_per_sec;
}

/**
 * @brief Main function
 */
int main() {
  std::cout << "RocksDB Read Performance Benchmark" << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Number of keys: " << NUM_KEYS << std::endl;
  std::cout << "  Value size: " << VALUE_SIZE << " bytes" << std::endl;
  std::cout << "  Number of threads: " << NUM_THREADS << std::endl;
  std::cout << "  Operations per thread: " << NUM_OPERATIONS_PER_THREAD
            << std::endl;
  std::cout << "==================================" << std::endl;

  // Setup the database with test data
  std::vector<std::string> keys = setup_database();

  // Vector to store benchmark results
  std::vector<std::pair<std::string, double>> results;

  // 1. Baseline: No optimizations
  {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.statistics = rocksdb::CreateDBStatistics();

    double ops_per_sec =
        run_benchmark(options, false, "Baseline (No Optimizations)", keys);
    results.emplace_back("Baseline", ops_per_sec);
  }

  // 2. Multi-get optimization
  {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.statistics = rocksdb::CreateDBStatistics();

    double ops_per_sec =
        run_benchmark(options, true, "Multi-Get Optimization", keys);
    results.emplace_back("Multi-Get", ops_per_sec);
  }

  // 3. Block cache optimization
  {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.statistics = rocksdb::CreateDBStatistics();

    // Set up the block cache (8MB)
    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(8 * 1024 * 1024);
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    double ops_per_sec =
        run_benchmark(options, false, "Block Cache (8MB)", keys);
    results.emplace_back("Block Cache (8MB)", ops_per_sec);
  }

  // 4. Prefix bloom filter optimization
  {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.statistics = rocksdb::CreateDBStatistics();
    options.prefix_extractor.reset(
        rocksdb::NewFixedPrefixTransform(PREFIX_LENGTH));

    rocksdb::BlockBasedTableOptions table_options;
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
    table_options.whole_key_filtering = true;
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    double ops_per_sec =
        run_benchmark(options, false, "Prefix Bloom Filter", keys);
    results.emplace_back("Prefix Bloom Filter", ops_per_sec);
  }

  // 5. Block cache size optimization
  for (size_t cache_size : BLOCK_CACHE_SIZES) {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.statistics = rocksdb::CreateDBStatistics();

    std::string cache_size_str =
        std::to_string(cache_size / (1024 * 1024)) + "MB";

    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(cache_size);
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    std::string test_name = "Block Cache Size (" + cache_size_str + ")";
    double ops_per_sec = run_benchmark(options, false, test_name, keys);
    results.emplace_back(test_name, ops_per_sec);
  }

  // Combined optimization: Block cache + Bloom filter + Multi-get
  {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.statistics = rocksdb::CreateDBStatistics();
    options.prefix_extractor.reset(
        rocksdb::NewFixedPrefixTransform(PREFIX_LENGTH));

    rocksdb::BlockBasedTableOptions table_options;
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
    table_options.block_cache =
        rocksdb::NewLRUCache(256 * 1024 * 1024); // 256MB cache
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    double ops_per_sec =
        run_benchmark(options, true, "Combined Optimizations", keys);
    results.emplace_back("Combined Optimizations", ops_per_sec);
  }

  // Print summary of results
  std::cout << "\nBenchmark Summary:" << std::endl;
  std::cout << "===================" << std::endl;

  // Find the baseline result
  double baseline_ops = 0;
  for (const auto &result : results) {
    if (result.first == "Baseline") {
      baseline_ops = result.second;
      break;
    }
  }

  // Print results with improvement percentage
  for (const auto &result : results) {
    double improvement =
        ((result.second - baseline_ops) / baseline_ops) * 100.0;
    std::cout << std::left << std::setw(30) << result.first << std::right
              << std::setw(12) << std::fixed << std::setprecision(2)
              << result.second << " ops/sec";

    if (result.first != "Baseline") {
      std::cout << " (" << (improvement >= 0 ? "+" : "") << std::fixed
                << std::setprecision(2) << improvement << "%)";
    }

    std::cout << std::endl;
  }

  // Clean up
  rocksdb::DestroyDB(DB_PATH, rocksdb::Options());

  return 0;
}
