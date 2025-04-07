/**
 * @file RocksDB scan test
 * @brief This file contains tests for scanning data from RocksDB
 *        It will try different optimizations and configurations to
 *        boost scan performance.
 */

/**
 * @brief Test schema
 * 1. Create a RocksDB database.
 * 2. Insert a large number of key-value pairs into the database.
 * 3. Close the database.
 * 4. Reopen the database.
 * 5. Perform multiple threads scan operation on the database without any
 * optimizations.
 * 6. Perform multiple threads scan operation with larger block cache size
 * 7. Enable cache_index_and_filter_blocks option
 * 8. Use ReadOptions::iterate_upper_bound to limit scan range
 * 9. Set ReadOptions::readahead_size (typically 256KB-1MB) to prefetch data
 * 10. Disable verify_checksums for scan-heavy workloads
 * 11. Scanning by prefix, configure prefix extractor,
 * options.prefix_extractor.reset(NewFixedPrefixTransform(prefix_length)); All
 * scan operations are performed in parallel using multiple threads with prefix
 */

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// Configuration parameters
constexpr size_t NUM_KEYS =
    100 * 1024 * 1024;             // Number of key-value pairs to insert
constexpr size_t VALUE_SIZE = 128; // Size of each value in bytes
constexpr int NUM_THREADS = 64;    // Number of threads for parallel scan
constexpr size_t SCAN_COUNT_PER_THREAD = 100; // Number of scans per thread
constexpr int PREFIX_LENGTH = 8; // Length of prefix for prefix scan
constexpr size_t DEFAULT_BLOCK_CACHE_SIZE = 8 * 1024 * 1024; // 8MB
constexpr size_t LARGE_BLOCK_CACHE_SIZE = 128 * 1024 * 1024; // 128MB
constexpr size_t READAHEAD_SIZE = 256 * 1024;                // 256KB readahead

// Print results with proper formatting
std::mutex print_mutex;
void print_result(const std::string &test_name, double duration_ms,
                  uint64_t record_count) {
  std::lock_guard<std::mutex> lock(print_mutex);
  std::cout << std::left << std::setw(50) << test_name << std::right
            << std::setw(10) << std::fixed << std::setprecision(2)
            << duration_ms << " ms " << record_count << std::endl;
}

// Generate a random string of specified length
std::string generate_random_string(size_t length) {
  static const char alphanum[] = "0123456789"
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                 "abcdefghijklmnopqrstuvwxyz";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);

  std::string result;
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result += alphanum[dis(gen)];
  }
  return result;
}

// Generate a key with a specific prefix and suffix
std::string generate_key_with_prefix(int prefix_id, int key_id) {
  return "prefix_" + std::to_string(prefix_id) + "_" + std::to_string(key_id);
}

// Class to manage the RocksDB scan test
class RocksDBScanTest {
private:
  std::string db_path_;
  std::shared_ptr<rocksdb::Cache> block_cache_;
  rocksdb::DB *db_ = nullptr;
  std::vector<std::string> prefixes_;
  std::atomic<bool> stop_flag_{false};
  static constexpr int NUM_PREFIXES =
      100; // Number of different prefixes to use

public:
  RocksDBScanTest(const std::string &path) : db_path_(path) {
    // Generate prefixes
    prefixes_.reserve(NUM_PREFIXES);
    for (int i = 0; i < NUM_PREFIXES; i++) {
      prefixes_.push_back(std::to_string(i));
    }
  }

  ~RocksDBScanTest() {
    if (db_) {
      delete db_;
    }
    // Clean up the database directory
    fs::remove_all(db_path_);
  }

  // Create and populate the database
  bool create_and_populate_db() {
    // Create directory if it doesn't exist
    try {
      if (fs::exists(db_path_)) {
        fs::remove_all(db_path_);
      }
      fs::create_directories(db_path_);
    } catch (const fs::filesystem_error &e) {
      std::cerr << "Error creating database directory: " << e.what()
                << std::endl;
      return false;
    }

    // Create database with default options
    rocksdb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    options.prefix_extractor.reset(
        rocksdb::NewFixedPrefixTransform(PREFIX_LENGTH));
    options.IncreaseParallelism(std::thread::hardware_concurrency());
    options.OptimizeLevelStyleCompaction();

    // Open the database
    rocksdb::Status status = rocksdb::DB::Open(options, db_path_, &db_);
    if (!status.ok()) {
      std::cerr << "Unable to create database: " << status.ToString()
                << std::endl;
      return false;
    }

    std::cout << "Populating database with " << NUM_KEYS
              << " key-value pairs..." << std::endl;

    // Generate and insert key-value pairs
    rocksdb::WriteOptions write_options;
    std::string value = generate_random_string(VALUE_SIZE);

    // Insert data with different prefixes
    for (int prefix_idx = 0; prefix_idx < NUM_PREFIXES; prefix_idx++) {
      rocksdb::WriteBatch batch;
      int keys_per_prefix = NUM_KEYS / NUM_PREFIXES;

      for (int i = 0; i < keys_per_prefix; i++) {
        std::string key = generate_key_with_prefix(prefix_idx, i);
        batch.Put(key, value);

        // Periodically write batch
        if ((i + 1) % 10000 == 0 || i == keys_per_prefix - 1) {
          status = db_->Write(write_options, &batch);
          if (!status.ok()) {
            std::cerr << "Write batch failed: " << status.ToString()
                      << std::endl;
            return false;
          }
          batch.Clear();

          // Print progress
          if ((i + 1) % 100000 == 0) {
            std::cout << "  Inserted " << (prefix_idx * keys_per_prefix + i + 1)
                      << " key-value pairs..." << std::endl;
          }
        }
      }
    }

    std::cout << "Database populated successfully." << std::endl;

    // Perform manual compaction to optimize read performance
    std::cout << "Performing manual compaction..." << std::endl;
    rocksdb::CompactRangeOptions compact_options;
    compact_options.bottommost_level_compaction =
        rocksdb::BottommostLevelCompaction::kForce;
    status = db_->CompactRange(compact_options, nullptr, nullptr);

    if (!status.ok()) {
      std::cerr << "Manual compaction failed: " << status.ToString()
                << std::endl;
    } else {
      std::cout << "Manual compaction completed successfully." << std::endl;
    }

    // Close the database
    delete db_;
    db_ = nullptr;

    return true;
  }

  // Reopen the database with given options
  bool reopen_db(const rocksdb::Options &options) {
    if (db_) {
      delete db_;
      db_ = nullptr;
    }

    rocksdb::Status status = rocksdb::DB::Open(options, db_path_, &db_);
    if (!status.ok()) {
      std::cerr << "Failed to open database: " << status.ToString()
                << std::endl;
      return false;
    }
    return true;
  }

  // Close the database
  void close_db() {
    if (db_) {
      delete db_;
      db_ = nullptr;
    }
  }

  // Create options with default settings
  rocksdb::Options create_default_options() {
    rocksdb::Options options;
    options.create_if_missing = false;
    options.IncreaseParallelism(std::thread::hardware_concurrency());
    options.OptimizeLevelStyleCompaction();
    return options;
  }

  // Create table options with specific block cache size
  rocksdb::BlockBasedTableOptions
  create_table_options(size_t block_cache_size) {
    rocksdb::BlockBasedTableOptions table_options;
    block_cache_ = rocksdb::NewLRUCache(block_cache_size);
    table_options.block_cache = block_cache_;
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
    return table_options;
  }

  // Run scan test with the given read options and description
  void run_scan_test(
      const std::string &test_name,
      const std::function<rocksdb::ReadOptions(int)> &read_options_creator) {
    // reopen the database
    reopen_db(create_default_options());

    std::vector<std::thread> threads;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Reset stop flag
    stop_flag_ = false;

    // Atomic variable to count records scanned
    std::atomic<uint64_t> record_count{0};

    // Start scanning threads
    for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++) {
      threads.emplace_back(
          [this, thread_id, &record_count, &read_options_creator]() {
            // Create a factory function that will generate fresh ReadOptions
            // with proper scope for any referenced variables
            auto read_options_factory = [thread_id, &read_options_creator]() {
              return read_options_creator(thread_id);
            };

            scan_worker(thread_id, record_count, read_options_factory);
          });
    }

    // Wait for threads to complete
    for (auto &thread : threads) {
      thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    print_result(test_name, static_cast<double>(duration), record_count.load());
    close_db();
  }

  // Worker function for scan operations
  void scan_worker(int thread_id, std::atomic<uint64_t> &record_count,
                   std::function<rocksdb::ReadOptions()> read_options_factory) {
    // Each thread operates on a subset of prefixes
    int prefixes_per_thread = NUM_PREFIXES / NUM_THREADS;
    int start_prefix = thread_id * prefixes_per_thread;
    int end_prefix = (thread_id == NUM_THREADS - 1)
                         ? NUM_PREFIXES
                         : start_prefix + prefixes_per_thread;
    // std::cout << "Thread " << thread_id << " scanning prefixes " <<
    // start_prefix
    //           << " to " << end_prefix - 1 << std::endl;

    for (int scan_count = 0; scan_count < SCAN_COUNT_PER_THREAD && !stop_flag_;
         scan_count++) {
      // Select a random prefix from this thread's range
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> prefix_dis(start_prefix, end_prefix - 1);
      int prefix_idx = prefix_dis(gen);

      // Create read options using the factory function
      // This ensures that any referenced variables are created in this scope
      rocksdb::ReadOptions read_options = read_options_factory();

      // Create iterator
      rocksdb::Iterator *iter = db_->NewIterator(read_options);
      std::string prefix = "prefix_" + prefixes_[prefix_idx] + "_";

      iter->Seek(prefix);

      // Scan all keys with this prefix
      int count = 0;
      // Limit the number of keys scanned to 10% of the total keys
      while (iter->Valid() && count < NUM_KEYS / 10) {
        rocksdb::Slice key = iter->key();
        if (!key.starts_with(prefix)) {
          break;
        }

        // Access the value (but don't do anything with it)
        rocksdb::Slice value = iter->value();
        count++;
        iter->Next();
      }

      if (!iter->status().ok()) {
        std::cerr << "Iterator error: " << iter->status().ToString()
                  << std::endl;
        stop_flag_ = true;
      }

      delete iter;
      record_count.fetch_add(count);
    }
  }

  // Run all tests with different optimizations
  void run_all_tests() {
    std::cout << std::left << std::setw(50) << "Test Name" << std::right
              << std::setw(10) << "Duration" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Test 1: No optimizations
    {
      std::cout << "Running test 1: No optimizations..." << std::endl;
      rocksdb::Options options = create_default_options();
      rocksdb::BlockBasedTableOptions table_options =
          create_table_options(DEFAULT_BLOCK_CACHE_SIZE);
      options.table_factory.reset(NewBlockBasedTableFactory(table_options));
      if (!reopen_db(options))
        return;

      run_scan_test("1. No optimizations",
                    [this](int thread_id) { return rocksdb::ReadOptions(); });
    }

    // Test 2: Larger block cache
    {
      std::cout << "Running test 2: Larger block cache..." << std::endl;
      rocksdb::Options options = create_default_options();
      rocksdb::BlockBasedTableOptions table_options =
          create_table_options(LARGE_BLOCK_CACHE_SIZE);
      options.table_factory.reset(NewBlockBasedTableFactory(table_options));
      if (!reopen_db(options))
        return;

      run_scan_test("2. Large block cache",
                    [this](int thread_id) { return rocksdb::ReadOptions(); });
    }

    // Test 3: Cache index and filter blocks
    {
      std::cout << "Running test 3: Cache index and filter blocks..."
                << std::endl;
      rocksdb::Options options = create_default_options();
      rocksdb::BlockBasedTableOptions table_options =
          create_table_options(LARGE_BLOCK_CACHE_SIZE);
      table_options.cache_index_and_filter_blocks = true;
      table_options.cache_index_and_filter_blocks_with_high_priority = true;
      options.table_factory.reset(NewBlockBasedTableFactory(table_options));
      if (!reopen_db(options))
        return;

      run_scan_test("3. Cache index and filter blocks",
                    [this](int thread_id) { return rocksdb::ReadOptions(); });
    }

    // Test 4: iterate_upper_bound to limit scan range
    {
      std::cout << "Running test 4: Using iterate_upper_bound..." << std::endl;
      rocksdb::Options options = create_default_options();
      if (!reopen_db(options))
        return;

      // Using thread_local to store upper bound strings
      run_scan_test("4. Using iterate_upper_bound", [this](int thread_id) {
        thread_local std::vector<std::string> upper_bounds;
        if (upper_bounds.empty()) {
          // Initialize on first use
          for (int i = 0; i < NUM_PREFIXES; i++) {
            std::string prefix = "prefix_" + prefixes_[i] + "_";
            upper_bounds.push_back(prefix +
                                   "\xFF"); // Upper bound just past the prefix
          }
        }

        rocksdb::ReadOptions read_options;
        // Each thread will use its subset of upper bounds
        int prefixes_per_thread = NUM_PREFIXES / NUM_THREADS;
        int start_prefix = thread_id * prefixes_per_thread;

        // We use a thread_local string for the upper bound
        // so it persists for the duration of the scan operation
        thread_local std::string upper_bound;
        upper_bound = upper_bounds[start_prefix];

        // Create a new slice using our thread_local string
        thread_local rocksdb::Slice upper_bounds_slice;
        upper_bounds_slice = rocksdb::Slice(upper_bound);

        read_options.iterate_upper_bound = &upper_bounds_slice;
        return read_options;
      });
    }

    // Test 5: readahead_size for prefetching
    {
      std::cout << "Running test 5: Using readahead_size..." << std::endl;
      rocksdb::Options options = create_default_options();
      if (!reopen_db(options))
        return;

      run_scan_test("5. Using readahead_size", [this](int thread_id) {
        rocksdb::ReadOptions read_options;
        read_options.readahead_size = READAHEAD_SIZE;
        return read_options;
      });
    }

    // Test 6: Disable verify_checksums
    {
      std::cout << "Running test 6: Disable verify_checksums..." << std::endl;
      rocksdb::Options options = create_default_options();
      if (!reopen_db(options))
        return;

      run_scan_test("6. Disable verify_checksums", [this](int thread_id) {
        rocksdb::ReadOptions read_options;
        read_options.verify_checksums = false;
        return read_options;
      });
    }

    // Test 7: Prefix scan with prefix extractor
    {
      std::cout << "Running test 7: Prefix scan with prefix extractor..."
                << std::endl;
      rocksdb::Options options = create_default_options();

      // Set prefix extractor
      options.prefix_extractor.reset(
          rocksdb::NewFixedPrefixTransform(PREFIX_LENGTH));
      if (!reopen_db(options))
        return;

      run_scan_test("7. Prefix scan with prefix extractor",
                    [this](int thread_id) {
                      rocksdb::ReadOptions read_options;
                      read_options.readahead_size = READAHEAD_SIZE;
                      return read_options;
                    });
    }

    // Test 8: Combine all optimizations
    {
      std::cout << "Running test 8: All optimizations combined..." << std::endl;
      rocksdb::Options options = create_default_options();
      rocksdb::BlockBasedTableOptions table_options =
          create_table_options(LARGE_BLOCK_CACHE_SIZE);
      table_options.cache_index_and_filter_blocks = true;
      table_options.cache_index_and_filter_blocks_with_high_priority = true;
      table_options.index_type =
          rocksdb::BlockBasedTableOptions::IndexType::kHashSearch;
      options.table_factory.reset(NewBlockBasedTableFactory(table_options));

      // Set prefix extractor
      options.prefix_extractor.reset(
          rocksdb::NewFixedPrefixTransform(PREFIX_LENGTH));
      if (!reopen_db(options))
        return;

      // Using thread_local to store upper bound strings
      run_scan_test("8. All optimizations combined", [this](int thread_id) {
        thread_local std::vector<std::string> upper_bounds;
        if (upper_bounds.empty()) {
          // Initialize on first use
          for (int i = 0; i < NUM_PREFIXES; i++) {
            std::string prefix = "prefix_" + prefixes_[i] + "_";
            upper_bounds.push_back(prefix +
                                   "\xFF"); // Upper bound just past the prefix
          }
        }

        rocksdb::ReadOptions read_options;
        read_options.verify_checksums = false;
        read_options.readahead_size = READAHEAD_SIZE;

        // Each thread will use its subset of upper bounds
        int prefixes_per_thread = NUM_PREFIXES / NUM_THREADS;
        int start_prefix = thread_id * prefixes_per_thread;

        // We use a thread_local string for the upper bound
        // so it persists for the duration of the scan operation
        thread_local std::string upper_bound;
        upper_bound = upper_bounds[start_prefix];

        // Create a new slice using our thread_local string
        thread_local rocksdb::Slice upper_bound_slice;
        upper_bound_slice = rocksdb::Slice(upper_bound);

        read_options.iterate_upper_bound = &upper_bound_slice;
        return read_options;
      });
    }
  }
};

int main() {
  std::cout << "RocksDB Scan Performance Test" << std::endl;
  std::cout << "============================" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Number of keys: " << NUM_KEYS << std::endl;
  std::cout << "  Value size: " << VALUE_SIZE << " bytes" << std::endl;
  std::cout << "  Number of threads: " << NUM_THREADS << std::endl;
  std::cout << "  Scans per thread: " << SCAN_COUNT_PER_THREAD << std::endl;
  std::cout << "  Default block cache: "
            << (DEFAULT_BLOCK_CACHE_SIZE / (1024 * 1024)) << " MB" << std::endl;
  std::cout << "  Large block cache: "
            << (LARGE_BLOCK_CACHE_SIZE / (1024 * 1024)) << " MB" << std::endl;
  std::cout << "  Readahead size: " << (READAHEAD_SIZE / 1024) << " KB"
            << std::endl;
  std::cout << std::endl;

  const std::string db_path = "/tmp/rocksdb_scan_test_db";
  RocksDBScanTest test(db_path);

  // Create and populate the database
  if (!test.create_and_populate_db()) {
    std::cerr << "Failed to create and populate the database." << std::endl;
    return 1;
  }

  // Run all tests
  test.run_all_tests();

  std::cout << std::endl;
  std::cout << "Scan performance test completed." << std::endl;
  return 0;
}
