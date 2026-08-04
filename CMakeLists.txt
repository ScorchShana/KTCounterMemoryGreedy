cmake_minimum_required(VERSION 3.10)
project(Tree LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # if compiler doesn't support C++20, then error

include(CheckIncludeFileCXX)

enable_testing()

# List implementation source files here. Replace or remove entries
# if the corresponding .cpp files do not exist.



set(CMAKE_CXX_FLAGS_DEBUG "-O3 -fomit-frame-pointer -g -march=native -flto=auto -fdiagnostics-color=always -Wpedantic -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fomit-frame-pointer -DNDEBUG -march=native -flto=auto -fdiagnostics-color=always -Wpedantic -Wall -Wextra")


file(GLOB_RECURSE SOURCES 
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)

#zlib-ng配置
set(BUILD_SHARED_LIBS OFF)
set(ZLIB_COMPAT ON)
set(WITH_NATIVE_INSTRUCTIONS ON)
set(WITH_OPTIM ON)
set(WITH_GZFILEOP ON)
#set(WITH_AVX512 OFF)
#set(WITH_AVX512VNNI OFF)
#set(WITH_VPCLMULQDQ OFF)
set(ZLIB_ENABLE_TESTS OFF)
set(ZLIB_ENABLE_EXAMPLES OFF)

add_subdirectory(third_party/zlib-ng)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} zlib)
target_link_libraries(${PROJECT_NAME} rt)
target_link_libraries(${PROJECT_NAME} pthread)

# 检查头文件
check_include_file_cxx(numa.h HAVE_NUMA_H)

if(HAVE_NUMA_H)
    message(STATUS "numa.h found, enabling NUMA support")
    target_link_libraries(${PROJECT_NAME} numa)
else()
    message(STATUS "libnuma not found, NUMA support disabled")
endif()


add_executable(concurrent_counting_hash_map_test tests/test_concurrent_counting_hash_map.cpp)
target_link_libraries(concurrent_counting_hash_map_test pthread)
if(HAVE_NUMA_H)
    message(STATUS "numa.h found, enabling NUMA support")
    target_link_libraries(concurrent_counting_hash_map_test numa)
else()
    message(STATUS "libnuma not found, NUMA support disabled")
endif()
add_test(NAME ConcurrentCountingHashMapTest COMMAND concurrent_counting_hash_map_test 1000 10)

add_executable(concurrent_map_test tests/test_concurrent_map.cpp)
target_link_libraries(concurrent_map_test pthread)
if(HAVE_NUMA_H)
    message(STATUS "numa.h found, enabling NUMA support")
    target_link_libraries(concurrent_map_test numa)
else()
    message(STATUS "libnuma not found, NUMA support disabled")
endif()
add_test(NAME ConcurrentMapTest COMMAND concurrent_map_test 1000 10)

add_executable(export_reader_test tests/test_export_reader.cpp)
target_link_libraries(export_reader_test rt)
target_link_libraries(export_reader_test pthread)
if(HAVE_NUMA_H)
    target_link_libraries(export_reader_test numa)
endif()
add_test(NAME ExportReaderTest COMMAND export_reader_test)

add_executable(final_drain_reader_test tests/test_final_drain_reader.cpp)
target_link_libraries(final_drain_reader_test rt)
target_link_libraries(final_drain_reader_test pthread)
if(HAVE_NUMA_H)
    target_link_libraries(final_drain_reader_test numa)
endif()
add_test(NAME FinalDrainReaderTest COMMAND final_drain_reader_test)

add_executable(histogram_tool tool/histogram_tool.cpp)
target_link_libraries(histogram_tool rt)
target_link_libraries(histogram_tool pthread)
if(HAVE_NUMA_H)
    target_link_libraries(histogram_tool numa)
endif()

add_executable(dump_tool tool/dump_tool.cpp)
target_link_libraries(dump_tool rt)
target_link_libraries(dump_tool pthread)
if(HAVE_NUMA_H)
    target_link_libraries(dump_tool numa)
endif()

add_executable(histogram_tool_test tests/test_histogram_tool.cpp)
target_link_libraries(histogram_tool_test rt)
target_link_libraries(histogram_tool_test pthread)
if(HAVE_NUMA_H)
    target_link_libraries(histogram_tool_test numa)
endif()
add_test(NAME HistogramToolTest COMMAND histogram_tool_test $<TARGET_FILE:histogram_tool>)

add_executable(low_frequency_query_thread_pool_test tests/test_low_frequency_query_thread_pool.cpp)
target_link_libraries(low_frequency_query_thread_pool_test pthread)
add_test(NAME LowFrequencyQueryThreadPoolTest COMMAND low_frequency_query_thread_pool_test)

add_executable(flat_concurrent_hash_map_avx2_test tests/test_flat_concurrent_hash_map.cpp)
target_compile_options(flat_concurrent_hash_map_avx2_test PRIVATE -mavx2)
target_link_libraries(flat_concurrent_hash_map_avx2_test pthread)
add_test(NAME FlatConcurrentHashMapAvx2Test COMMAND flat_concurrent_hash_map_avx2_test)

add_executable(flat_concurrent_hash_map_sse_test tests/test_flat_concurrent_hash_map.cpp)
target_compile_options(flat_concurrent_hash_map_sse_test PRIVATE -mno-avx2 -msse4.2)
target_link_libraries(flat_concurrent_hash_map_sse_test pthread)
add_test(NAME FlatConcurrentHashMapSseTest COMMAND flat_concurrent_hash_map_sse_test)

add_executable(flat_concurrent_hash_map_scalar_test tests/test_flat_concurrent_hash_map.cpp)
target_compile_options(flat_concurrent_hash_map_scalar_test PRIVATE -mno-avx2 -mno-sse4.2)
target_link_libraries(flat_concurrent_hash_map_scalar_test pthread)
add_test(NAME FlatConcurrentHashMapScalarTest COMMAND flat_concurrent_hash_map_scalar_test)

add_executable(benchmark_final_drain_reader_test tests/benchmark_final_drain_reader.cpp)
target_link_libraries(benchmark_final_drain_reader_test pthread)
target_link_libraries(benchmark_final_drain_reader_test rt)

add_executable(benchmark_export_reader_test tests/benchmark_export_reader.cpp)
target_link_libraries(benchmark_export_reader_test pthread)
target_link_libraries(benchmark_export_reader_test rt)

add_executable(test_and_benckmark_export_record_radix_sort tests/test_and_benckmark_export_record_radix_sort.cpp)