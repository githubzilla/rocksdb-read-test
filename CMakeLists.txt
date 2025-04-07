cmake_minimum_required(VERSION 3.10)
project(rocksdb_read_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(ROCKSDB_INCLUDE_PATH NAMES rocksdb/db.h)
if (NOT ROCKSDB_INCLUDE_PATH)
  message(FATAL_ERROR "Fail to find RocksDB include path")
endif ()
message(STATUS "ROCKSDB_INCLUDE_PATH: ${ROCKSDB_INCLUDE_PATH}")

# Find RocksDB package
find_library(ROCKSDB_LIB NAMES rocksdb)
if (NOT ROCKSDB_LIB)
    message(FATAL_ERROR "Fail to find RocksDB lib path")
endif ()
message(STATUS "ROCKSDB_LIB: ${ROCKSDB_LIB}")

# Add the executable
add_executable(rocksdb_read_test rocksdb_read_test.cpp)
add_executable(rocksdb_scan_test rocksdb_scan_test.cpp)

# Link the executable to required libraries
target_link_libraries(rocksdb_read_test PRIVATE ${ROCKSDB_LIB} pthread)
target_link_libraries(rocksdb_scan_test PRIVATE ${ROCKSDB_LIB} pthread)

# Set compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

