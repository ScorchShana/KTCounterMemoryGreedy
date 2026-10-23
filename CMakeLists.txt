cmake_minimum_required(VERSION 3.10)
project(Tree LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # if compiler doesn't support C++20, then error

include(CheckIncludeFileCXX)

enable_testing()
option(TEST_MODE "Enable diagnostic statistics and output for Tree" OFF)

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
if(TEST_MODE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TEST_MODE)
endif()

target_link_libraries(${PROJECT_NAME} zlib)
target_link_libraries(${PROJECT_NAME} rt)
target_link_libraries(${PROJECT_NAME} pthread)
target_link_libraries(${PROJECT_NAME} atomic)

# 检查头文件
check_include_file_cxx(numa.h HAVE_NUMA_H)
find_library(NUMA_LIBRARY numa)

if(HAVE_NUMA_H)
    message(STATUS "numa.h found, enabling NUMA support")
    target_link_libraries(${PROJECT_NAME} numa)
else()
    message(STATUS "libnuma not found, NUMA support disabled")
endif()


foreach(SIMD IN ITEMS scalar sse avx2)
    add_executable(counting_hash_map_${SIMD}_test tests/test_counting_hash_map.cpp)
    if(SIMD STREQUAL "avx2")
        target_compile_options(counting_hash_map_${SIMD}_test PRIVATE -mavx2)
    elseif(SIMD STREQUAL "sse")
        target_compile_options(counting_hash_map_${SIMD}_test PRIVATE -mno-avx -mno-avx2 -mno-sse4.2 -msse2)
    else()
        target_compile_options(counting_hash_map_${SIMD}_test PRIVATE -mno-avx -mno-avx2 -mno-sse2)
    endif()
    target_link_libraries(counting_hash_map_${SIMD}_test pthread)
    add_test(NAME CountingHashMap_${SIMD} COMMAND counting_hash_map_${SIMD}_test)
    set_tests_properties(CountingHashMap_${SIMD} PROPERTIES TIMEOUT 30)
endforeach()

add_executable(concurrent_open_address_hash_map_test tests/test_concurrent_open_address_hash_map.cpp)
target_link_libraries(concurrent_open_address_hash_map_test pthread rt)
if(NUMA_LIBRARY)
    target_link_libraries(concurrent_open_address_hash_map_test ${NUMA_LIBRARY})
endif()
add_test(NAME ConcurrentOpenAddressHashMapTest COMMAND concurrent_open_address_hash_map_test)
set_tests_properties(ConcurrentOpenAddressHashMapTest PROPERTIES TIMEOUT 120)

option(ENABLE_TSAN_TESTS "Build ThreadSanitizer tests" OFF)
if(ENABLE_TSAN_TESTS)
    add_executable(concurrent_open_address_hash_map_tsan_test tests/test_concurrent_open_address_hash_map.cpp)
    target_compile_options(concurrent_open_address_hash_map_tsan_test PRIVATE
        -O3
        -g
        -fsanitize=thread
        -fno-omit-frame-pointer
        -fno-lto)
    set_property(TARGET concurrent_open_address_hash_map_tsan_test
        APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=thread -fno-lto")
    target_link_libraries(concurrent_open_address_hash_map_tsan_test pthread rt)
    if(NUMA_LIBRARY)
        target_link_libraries(concurrent_open_address_hash_map_tsan_test ${NUMA_LIBRARY})
    endif()
    find_program(SETARCH_EXECUTABLE setarch)
    if(SETARCH_EXECUTABLE)
        add_test(NAME ConcurrentOpenAddressHashMapTsanTest
            COMMAND ${SETARCH_EXECUTABLE} ${CMAKE_SYSTEM_PROCESSOR} -R
                $<TARGET_FILE:concurrent_open_address_hash_map_tsan_test>)
    else()
        add_test(NAME ConcurrentOpenAddressHashMapTsanTest
            COMMAND concurrent_open_address_hash_map_tsan_test)
    endif()
    set_tests_properties(ConcurrentOpenAddressHashMapTsanTest PROPERTIES
        LABELS "tsan"
        TIMEOUT 180
        ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1")
endif()

add_executable(export_reader_test tests/test_export_reader.cpp)
target_link_libraries(export_reader_test rt)
target_link_libraries(export_reader_test pthread)
if(HAVE_NUMA_H)
    target_link_libraries(export_reader_test numa)
endif()
add_test(NAME ExportReaderTest COMMAND export_reader_test)

add_executable(reader_thread_pool_test tests/test_reader_thread_pool.cpp)
target_link_libraries(reader_thread_pool_test zlib rt pthread atomic)
if(HAVE_NUMA_H)
    target_link_libraries(reader_thread_pool_test numa)
endif()
add_test(NAME ReaderThreadPoolTest COMMAND reader_thread_pool_test)

add_executable(fastq_pre_reader_test tests/test_fastq_pre_reader.cpp)
target_link_libraries(fastq_pre_reader_test zlib rt pthread atomic)
add_test(NAME FastqPreReaderTest COMMAND fastq_pre_reader_test)
set_tests_properties(FastqPreReaderTest PROPERTIES TIMEOUT 30)

add_executable(final_drain_reader_test tests/test_final_drain_reader.cpp)
target_link_libraries(final_drain_reader_test rt)
target_link_libraries(final_drain_reader_test pthread)
if(HAVE_NUMA_H)
    target_link_libraries(final_drain_reader_test numa)
endif()
add_test(NAME FinalDrainReaderTest COMMAND final_drain_reader_test)

add_executable(histogram_tool tool/histogram_tool.cpp tool/HistogramRunner.cpp)
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
set_tests_properties(HistogramToolTest PROPERTIES TIMEOUT 180)

add_executable(compact_record_reader_test tests/test_compact_record_reader.cpp)
target_link_libraries(compact_record_reader_test pthread rt)
add_test(NAME CompactRecordReaderTest COMMAND compact_record_reader_test)
set_tests_properties(CompactRecordReaderTest PROPERTIES TIMEOUT 60)

add_executable(histogram_allocation_failure_test tests/test_histogram_allocation_failure.cpp)
target_link_libraries(histogram_allocation_failure_test pthread dl)
add_test(NAME HistogramMappingFailureTest COMMAND histogram_allocation_failure_test mappings)
add_test(NAME HistogramThreadFailureTest COMMAND histogram_allocation_failure_test threads)
set_tests_properties(HistogramMappingFailureTest HistogramThreadFailureTest PROPERTIES TIMEOUT 30)

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

foreach(RING_TEST IN ITEMS bloom_repeat classifier_first_batch concurrent_first_block_ring)
    add_executable(${RING_TEST}_test tests/test_${RING_TEST}.cpp)
    target_link_libraries(${RING_TEST}_test pthread rt atomic)
    if(NUMA_LIBRARY)
        target_link_libraries(${RING_TEST}_test ${NUMA_LIBRARY})
    endif()
endforeach()
add_executable(low_frequency_io_test tests/test_low_frequency_io.cpp)
target_link_libraries(low_frequency_io_test pthread rt atomic)
if(NUMA_LIBRARY)
    target_link_libraries(low_frequency_io_test ${NUMA_LIBRARY})
endif()
add_test(NAME LowFrequencyIOTest COMMAND low_frequency_io_test)
set_tests_properties(LowFrequencyIOTest PROPERTIES TIMEOUT 30)
target_compile_definitions(classifier_first_batch_test PRIVATE TEST_MODE)
add_test(NAME ConcurrentFirstBlockRingTest COMMAND concurrent_first_block_ring_test)
set_tests_properties(ConcurrentFirstBlockRingTest PROPERTIES TIMEOUT 60)
if(ENABLE_TSAN_TESTS)
    add_executable(concurrent_first_block_ring_tsan_test tests/test_concurrent_first_block_ring.cpp)
    target_compile_options(concurrent_first_block_ring_tsan_test PRIVATE
        -O1 -g -fsanitize=thread -fno-omit-frame-pointer -fno-lto)
    set_property(TARGET concurrent_first_block_ring_tsan_test
        APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=thread -fno-lto")
    target_link_libraries(concurrent_first_block_ring_tsan_test pthread rt atomic)
    if(NUMA_LIBRARY)
        target_link_libraries(concurrent_first_block_ring_tsan_test ${NUMA_LIBRARY})
    endif()
    if(SETARCH_EXECUTABLE)
        add_test(NAME ConcurrentFirstBlockRingTsanTest
            COMMAND ${SETARCH_EXECUTABLE} ${CMAKE_SYSTEM_PROCESSOR} -R
                $<TARGET_FILE:concurrent_first_block_ring_tsan_test> --no-death)
    else()
        add_test(NAME ConcurrentFirstBlockRingTsanTest COMMAND concurrent_first_block_ring_tsan_test --no-death)
    endif()
    set_tests_properties(ConcurrentFirstBlockRingTsanTest PROPERTIES
        LABELS "tsan" TIMEOUT 180 ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1")
endif()
add_test(NAME BloomRepeatTest COMMAND bloom_repeat_test)
add_test(NAME ClassifierFirstBatchTest COMMAND classifier_first_batch_test)
set_tests_properties(BloomRepeatTest ClassifierFirstBatchTest PROPERTIES TIMEOUT 60)
add_executable(get_kmer_test tests/test_get_kmer.cpp)
add_test(NAME GetKmerTest COMMAND get_kmer_test)
set_tests_properties(GetKmerTest PROPERTIES TIMEOUT 30)
find_program(PYTHON3_EXECUTABLE NAMES python3)
if(PYTHON3_EXECUTABLE)
    add_test(NAME ClassifierEndToEndTest
        COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_classifier_end_to_end.py
            $<TARGET_FILE:Tree>)
    set_tests_properties(ClassifierEndToEndTest PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endif()

add_executable(benchmark_final_drain_reader_test tests/benchmark_final_drain_reader.cpp)
target_link_libraries(benchmark_final_drain_reader_test pthread)
target_link_libraries(benchmark_final_drain_reader_test rt)

add_executable(benchmark_export_reader_test tests/benchmark_export_reader.cpp)
target_link_libraries(benchmark_export_reader_test pthread)
target_link_libraries(benchmark_export_reader_test rt)

add_executable(test_and_benckmark_export_record_radix_sort tests/test_and_benckmark_export_record_radix_sort.cpp)

add_executable(benchmark_gzip_streamer_test tests/benchmark_gzip_streamer.cpp)
target_link_libraries(benchmark_gzip_streamer_test zlib)
target_link_libraries(benchmark_gzip_streamer_test pthread)
target_link_libraries(benchmark_gzip_streamer_test rt)

add_executable(benchmark_gzip_base_extraction_test tests/benchmark_gzip_base_extraction.cpp)
target_link_libraries(benchmark_gzip_base_extraction_test zlib)
target_link_libraries(benchmark_gzip_base_extraction_test pthread)
target_link_libraries(benchmark_gzip_base_extraction_test rt)

add_executable(memory_budget_test tests/test_memory_budget.cpp)
target_link_libraries(memory_budget_test pthread rt atomic)
if(NUMA_LIBRARY)
    target_link_libraries(memory_budget_test ${NUMA_LIBRARY})
endif()
add_test(NAME MemoryBudgetTest COMMAND memory_budget_test)
set_tests_properties(MemoryBudgetTest PROPERTIES TIMEOUT 30)

foreach(POOL_TEST IN ITEMS concurrent_memory_pool run_state first_full_segments first_pool_lifecycle tree_allocation_failure first_spill_writer large_memory_calculations large_memory_mapping file_arena first_memory_scan)
    add_executable(${POOL_TEST}_test tests/test_${POOL_TEST}.cpp)
    target_link_libraries(${POOL_TEST}_test pthread rt atomic)
    if(NUMA_LIBRARY)
        target_link_libraries(${POOL_TEST}_test ${NUMA_LIBRARY})
    endif()
endforeach()
target_compile_definitions(first_full_segments_test PRIVATE TEST_MODE)
target_compile_definitions(first_memory_scan_test PRIVATE TEST_MODE)
target_compile_definitions(first_spill_writer_test PRIVATE TEST_MODE)
target_compile_definitions(run_state_test PRIVATE TEST_MODE)
add_test(NAME ConcurrentMemoryPoolTest COMMAND concurrent_memory_pool_test)
add_test(NAME FileArenaTest COMMAND file_arena_test)
set_tests_properties(FileArenaTest PROPERTIES TIMEOUT 60)
add_test(NAME FirstMemoryScanTest COMMAND first_memory_scan_test)
set_tests_properties(FirstMemoryScanTest PROPERTIES TIMEOUT 30)
add_test(NAME LargeMemoryCalculationsTest COMMAND large_memory_calculations_test)
set_tests_properties(LargeMemoryCalculationsTest PROPERTIES TIMEOUT 30)
add_test(NAME LargeMemoryMappingTest COMMAND large_memory_mapping_test)
set_tests_properties(LargeMemoryMappingTest PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
add_test(NAME RunStateTest COMMAND run_state_test)
add_test(NAME FirstFullSegmentsTest COMMAND first_full_segments_test)
add_test(NAME FirstPoolLifecycleTest COMMAND first_pool_lifecycle_test)
set_tests_properties(ConcurrentMemoryPoolTest RunStateTest FirstFullSegmentsTest FirstPoolLifecycleTest PROPERTIES TIMEOUT 30)

add_test(NAME TreeAllocationFailureTest COMMAND tree_allocation_failure_test)
set_tests_properties(TreeAllocationFailureTest PROPERTIES TIMEOUT 30)

add_test(NAME FirstSpillWriterTest COMMAND first_spill_writer_test)
set_tests_properties(FirstSpillWriterTest PROPERTIES TIMEOUT 60)
if(PYTHON3_EXECUTABLE)
    add_test(NAME FirstPoolEndToEndTest COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_first_pool_end_to_end.py $<TARGET_FILE:Tree>)
    set_tests_properties(FirstPoolEndToEndTest PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
endif()


if(PYTHON3_EXECUTABLE)
    add_test(NAME FirstPoolFailureTest COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_first_pool_failures.py $<TARGET_FILE:Tree>)
    set_tests_properties(FirstPoolFailureTest PROPERTIES TIMEOUT 75 RUN_SERIAL TRUE)
endif()
